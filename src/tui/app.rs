use crate::claude::LogEntry;
use crate::debug_log;
use crate::error::{AppError, Result};
use crate::history::grouping::{ProjectGroup, group_by_project_path};
use crate::history::{
    Conversation, LoaderMessage, ProviderKind, format_short_name_from_path,
    process_conversation_file,
};
use crate::providers::Provider;
use crate::tui::search::{self, SearchableConversation};
use crate::tui::ui;
use crate::tui::viewer::ToolDisplayMode;
use chrono::Local;
use crossterm::cursor::{Hide, Show};
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::terminal::{self, EnterAlternateScreen, LeaveAlternateScreen};
use ratatui::prelude::*;
use std::cell::Cell;
use std::io::{self, Stdout};
use std::path::PathBuf;
use std::sync::mpsc::Receiver;
use std::time::Duration;

/// Returns (label, color, dim_color) for a provider
fn provider_theme(kind: &ProviderKind) -> (String, (u8, u8, u8), (u8, u8, u8)) {
    match kind {
        ProviderKind::Claude => ("Claude".to_string(), (218, 119, 86), (170, 93, 67)),
        ProviderKind::Cursor => ("Cursor IDE".to_string(), (180, 130, 230), (140, 100, 180)),
        ProviderKind::CursorAgent => ("Cursor Agent".to_string(), (94, 184, 255), (72, 140, 194)),
    }
}

/// Result of running the TUI
pub enum Action {
    Select(PathBuf),
    Delete(PathBuf),
    Resume(PathBuf),
    Quit,
}

/// Dialog overlay mode (for confirmations, menus)
#[derive(Clone, Debug, PartialEq)]
pub enum DialogMode {
    /// No dialog shown
    None,
    /// Confirming deletion of the selected conversation
    ConfirmDelete,
    /// Export menu (save to file)
    ExportMenu { selected: usize },
    /// Yank menu (copy to clipboard)
    YankMenu { selected: usize },
    /// Help overlay showing keyboard shortcuts
    Help,
}

/// Export format options for menus
const EXPORT_OPTIONS: [&str; 4] = [
    "Ledger (formatted)",
    "Plain text",
    "Markdown",
    "JSONL (raw)",
];

/// Main application mode
#[derive(Clone, Debug)]
pub enum AppMode {
    /// List mode - browsing conversations
    List,
    /// View mode - reading a conversation
    View(ViewState),
}

/// Layout of the conversation list.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ListViewMode {
    /// Original flat list, sorted by recency.
    Flat,
    /// Hierarchical dashboard grouped by project path.
    Grouped,
}

/// Visible row in the grouped dashboard. Built from `groups` + `expanded_groups`
/// every time the list state changes; selection is an index into this vector.
#[derive(Clone, Debug)]
pub enum Row {
    /// Group header. `group_idx` indexes into `App::groups`.
    Header { group_idx: usize },
    /// Conversation row. `conv_idx` indexes into `App::conversations`.
    Conversation { group_idx: usize, conv_idx: usize },
}

/// Which pane currently receives keystrokes (Grouped mode).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PaneFocus {
    List,
    Preview,
}

/// Cache key for the preview render: invalidates when toggles change.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct PreviewKey {
    path: PathBuf,
    width: usize,
    tool_display: ToolDisplayMode,
    show_thinking: bool,
}

/// State for the conversation viewer
#[derive(Clone, Debug)]
pub struct ViewState {
    /// Path to the conversation file (stable identity)
    pub conversation_path: PathBuf,
    /// Current scroll position (line offset)
    pub scroll_offset: usize,
    /// Pre-rendered conversation lines
    pub rendered_lines: Vec<RenderedLine>,
    /// Total content height in lines
    pub total_lines: usize,
    /// Tool display mode (hidden/truncated/full)
    pub tool_display: ToolDisplayMode,
    /// Whether to show thinking blocks
    pub show_thinking: bool,
    /// Whether to show timing information (timestamps + durations)
    pub show_timing: bool,
    /// Content width used for rendering (for resize detection)
    pub content_width: usize,
    /// Search mode state
    pub search_mode: ViewSearchMode,
    /// Current search query
    pub search_query: String,
    /// Line indices with matches
    pub search_matches: Vec<usize>,
    /// Current match index
    pub current_match: usize,
    /// Cached log entries to avoid re-reading from disk on toggle
    pub cached_entries: Vec<LogEntry>,
}

/// Search mode within view
#[derive(Clone, Debug, PartialEq, Default)]
pub enum ViewSearchMode {
    #[default]
    Off,
    /// Typing search query
    Typing,
    /// Search active, navigating results
    Active,
}

/// A single rendered line with its spans
#[derive(Clone, Debug)]
pub struct RenderedLine {
    pub spans: Vec<(String, LineStyle)>,
}

/// Style information for a span
#[derive(Clone, Debug, Default)]
pub struct LineStyle {
    pub fg: Option<(u8, u8, u8)>,
    pub bold: bool,
    pub dimmed: bool,
    pub italic: bool,
}

/// Loading state for the TUI
#[derive(Clone, Debug)]
pub enum LoadingState {
    /// Still loading conversations
    Loading { loaded: usize },
    /// All conversations loaded and ready
    Ready,
}

/// App state
pub struct App {
    /// All loaded conversations
    conversations: Vec<Conversation>,
    /// Precomputed search data
    searchable: Vec<SearchableConversation>,
    /// Indices into conversations, sorted by current score
    filtered: Vec<usize>,
    /// Currently selected index into filtered (None if no results)
    selected: Option<usize>,
    /// Current search query
    query: String,
    /// Parsed and normalized query words (cached for render performance)
    query_words: Vec<String>,
    /// Cursor position in query (character index, not byte)
    cursor_pos: usize,
    /// Whether to use relative time display
    use_relative_time: bool,
    /// Loading state
    loading_state: LoadingState,
    /// Current dialog overlay (confirm, menu)
    dialog_mode: DialogMode,
    /// Main app mode (list or view)
    app_mode: AppMode,
    /// Status message with timestamp for auto-clear
    status_message: Option<(String, std::time::Instant)>,
    /// Persistent view setting: tool display mode
    tool_display: ToolDisplayMode,
    /// Persistent view setting: whether to show thinking blocks
    show_thinking: bool,
    /// Persistent view setting: whether to show timing information
    show_timing: bool,
    /// Whether the app is running in single file mode (direct input, no list)
    single_file_mode: bool,
    /// Previous search query for incremental filtering
    previous_query: String,
    /// Whether to show conversations whose project directory no longer exists
    show_deleted_projects: bool,
    /// Layout of the conversation list.
    view_mode: ListViewMode,
    /// Project groups derived from conversations (Grouped mode only).
    groups: Vec<ProjectGroup>,
    /// Canonical paths of currently expanded groups.
    expanded_groups: std::collections::HashSet<PathBuf>,
    /// Flattened, expansion-aware view of `groups` for rendering and selection.
    rows: Vec<Row>,
    /// Selected index into `rows` (Grouped mode). None when no rows.
    selected_row: Option<usize>,
    /// Whether the right preview pane is shown (Grouped mode).
    preview_visible: bool,
    /// Which pane currently has focus.
    pane_focus: PaneFocus,
    /// Vertical scroll offset for the preview pane.
    preview_scroll: usize,
    /// Last viewport height used to render the preview pane. Set during render
    /// so scroll handlers can clamp without needing the layout reference.
    preview_viewport_height: Cell<usize>,
    /// Cached rendered preview lines + key (path/width/toggles).
    preview_cache: Option<(PreviewKey, Vec<RenderedLine>)>,
}

impl App {
    /// Create a new app with all conversations pre-loaded (existing behavior)
    pub fn new(
        mut conversations: Vec<Conversation>,
        use_relative_time: bool,
        tool_display: ToolDisplayMode,
        show_thinking: bool,
        show_deleted_projects: bool,
    ) -> Self {
        if !show_deleted_projects {
            conversations.retain(|c| c.project_path.as_ref().map_or(true, |p| p.exists()));
        }
        let searchable = search::precompute_search_text(&mut conversations);
        let filtered: Vec<usize> = (0..conversations.len()).collect();
        let selected = if filtered.is_empty() { None } else { Some(0) };

        let mut app = Self {
            conversations,
            searchable,
            filtered,
            selected,
            query: String::new(),
            query_words: Vec::new(),
            cursor_pos: 0,
            use_relative_time,
            loading_state: LoadingState::Ready,
            dialog_mode: DialogMode::None,
            app_mode: AppMode::List,
            status_message: None,
            tool_display,
            show_thinking,
            show_timing: false,
            single_file_mode: false,
            previous_query: String::new(),
            show_deleted_projects,
            view_mode: ListViewMode::Grouped,
            groups: Vec::new(),
            expanded_groups: std::collections::HashSet::new(),
            rows: Vec::new(),
            selected_row: None,
            preview_visible: true,
            pane_focus: PaneFocus::List,
            preview_scroll: 0,
            preview_viewport_height: Cell::new(0),
            preview_cache: None,
        };
        app.rebuild_groups();
        app
    }

    /// Create a new app in loading state
    pub fn new_loading(
        use_relative_time: bool,
        tool_display: ToolDisplayMode,
        show_thinking: bool,
        show_deleted_projects: bool,
    ) -> Self {
        Self {
            conversations: Vec::new(),
            searchable: Vec::new(),
            filtered: Vec::new(),
            selected: None,
            query: String::new(),
            query_words: Vec::new(),
            cursor_pos: 0,
            use_relative_time,
            loading_state: LoadingState::Loading { loaded: 0 },
            dialog_mode: DialogMode::None,
            app_mode: AppMode::List,
            status_message: None,
            tool_display,
            show_thinking,
            show_timing: false,
            single_file_mode: false,
            previous_query: String::new(),
            show_deleted_projects,
            view_mode: ListViewMode::Grouped,
            groups: Vec::new(),
            expanded_groups: std::collections::HashSet::new(),
            rows: Vec::new(),
            selected_row: None,
            preview_visible: true,
            pane_focus: PaneFocus::List,
            preview_scroll: 0,
            preview_viewport_height: Cell::new(0),
            preview_cache: None,
        }
    }

    /// Create a new app for viewing a single file directly
    pub fn new_single_file(
        path: PathBuf,
        use_relative_time: bool,
        tool_display: ToolDisplayMode,
        show_thinking: bool,
    ) -> Self {
        // Parse using the same parser as the main list
        let modified = std::fs::metadata(&path).and_then(|m| m.modified()).ok();

        let mut conversations = Vec::new();
        let mut filtered = Vec::new();
        let mut selected = None;

        if let Ok(Some(mut conv)) = process_conversation_file(path.clone(), false, modified, None) {
            // Set project_name the same way as the loader does
            let project_path = conv.cwd.clone().unwrap_or_else(|| path.clone());
            conv.project_name = Some(format_short_name_from_path(&project_path));

            conversations.push(conv);
            filtered.push(0);
            selected = Some(0);
        }

        Self {
            conversations,
            searchable: Vec::new(),
            filtered,
            selected,
            query: String::new(),
            query_words: Vec::new(),
            cursor_pos: 0,
            use_relative_time,
            loading_state: LoadingState::Ready,
            dialog_mode: DialogMode::None,
            app_mode: AppMode::View(ViewState {
                conversation_path: path,
                scroll_offset: 0,
                rendered_lines: Vec::new(),
                total_lines: 0,
                tool_display,
                show_thinking,
                show_timing: false,
                content_width: 0,
                search_mode: ViewSearchMode::Off,
                search_query: String::new(),
                search_matches: Vec::new(),
                current_match: 0,
                cached_entries: Vec::new(),
            }),
            status_message: None,
            tool_display,
            show_thinking,
            show_timing: false,
            single_file_mode: true,
            previous_query: String::new(),
            show_deleted_projects: true,
            view_mode: ListViewMode::Flat,
            groups: Vec::new(),
            expanded_groups: std::collections::HashSet::new(),
            rows: Vec::new(),
            selected_row: None,
            preview_visible: false,
            pane_focus: PaneFocus::List,
            preview_scroll: 0,
            preview_viewport_height: Cell::new(0),
            preview_cache: None,
        }
    }

    /// Append a batch of conversations during loading.
    /// Defers global sort and search precompute until finish_loading to avoid
    /// repeated O(n log n) work while provider batches are still arriving.
    pub fn append_conversations(&mut self, mut new_convs: Vec<Conversation>) {
        if !self.show_deleted_projects {
            new_convs.retain(|c| c.project_path.as_ref().map_or(true, |p| p.exists()));
        }
        self.conversations.extend(new_convs);

        // Rebuild filtered as sequential indices (no search during loading)
        self.filtered = (0..self.conversations.len()).collect();

        // Select first item if nothing selected yet
        if self.selected.is_none() && !self.filtered.is_empty() {
            self.selected = Some(0);
        }

        // Update loading count
        self.loading_state = LoadingState::Loading {
            loaded: self.conversations.len(),
        };

        if self.view_mode == ListViewMode::Grouped {
            self.rebuild_groups();
        }
    }

    /// Mark loading as complete: sort, precompute search, and transition to Ready
    pub fn finish_loading(&mut self) {
        // Sort all conversations by timestamp (newest first)
        self.conversations
            .sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

        // Reindex after sorting
        for (idx, conv) in self.conversations.iter_mut().enumerate() {
            conv.index = idx;
        }

        // Now precompute search text (only once, at the end)
        self.searchable = search::precompute_search_text(&mut self.conversations);

        self.loading_state = LoadingState::Ready;

        // Apply any query that was typed during loading
        if self.query.is_empty() {
            // Reset filtered to all indices
            self.filtered = (0..self.conversations.len()).collect();
            self.selected = if self.filtered.is_empty() {
                None
            } else {
                Some(0)
            };
        } else {
            // User typed during loading, apply the filter now
            self.update_filter();
        }

        if self.view_mode == ListViewMode::Grouped {
            self.rebuild_groups();
        }
    }

    /// Consume the app and return its conversations
    pub fn into_conversations(self) -> Vec<Conversation> {
        self.conversations
    }

    pub fn loading_state(&self) -> &LoadingState {
        &self.loading_state
    }

    pub fn is_loading(&self) -> bool {
        matches!(self.loading_state, LoadingState::Loading { .. })
    }

    /// Refresh the cached query words from the current query
    fn refresh_query_words(&mut self) {
        let query_normalized = search::normalize_for_search(self.query.trim());
        self.query_words = query_normalized
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();
    }

    /// Update filtered results based on current query
    fn update_filter(&mut self) {
        let now = Local::now();
        // When the new query extends the previous one, only rescore the already-filtered subset
        let narrow_hint =
            if !self.previous_query.is_empty() && self.query.starts_with(&self.previous_query) {
                // Move filtered out to avoid clone; search() will produce the new value
                Some(std::mem::take(&mut self.filtered))
            } else {
                None
            };
        self.filtered = search::search(
            &self.conversations,
            &self.searchable,
            &self.query,
            now,
            narrow_hint.as_deref(),
        );
        self.previous_query.clone_from(&self.query);
        self.selected = if self.filtered.is_empty() {
            None
        } else {
            Some(0)
        };

        // Cache parsed query words for render performance
        self.refresh_query_words();

        if self.view_mode == ListViewMode::Grouped {
            self.rebuild_rows();
            // Reset selection to first visible row when query changes.
            self.selected_row = if self.rows.is_empty() { None } else { Some(0) };
        }
    }

    /// Recompute groups and rebuild the visible row vector.
    fn rebuild_groups(&mut self) {
        self.groups = group_by_project_path(&self.conversations);

        // First-time expansion seed: top group expanded so the user sees content.
        if self.expanded_groups.is_empty()
            && let Some(first) = self.groups.first()
            && let Some(path) = first.canonical_path.clone()
        {
            self.expanded_groups.insert(path);
        }

        self.rebuild_rows();
    }

    /// Build the flat list of visible rows from `groups`, `filtered`, and `expanded_groups`.
    fn rebuild_rows(&mut self) {
        let mut rows = Vec::with_capacity(self.groups.len() * 2);

        // When a query is active we want only conversations that survived filtering;
        // auto-expand groups that have matches and hide groups with none.
        let query_active = !self.query.is_empty();
        let allowed: Option<std::collections::HashSet<usize>> = if query_active {
            Some(self.filtered.iter().copied().collect())
        } else {
            None
        };

        for (gi, group) in self.groups.iter().enumerate() {
            let visible_children: Vec<usize> = if let Some(ref allowed) = allowed {
                group
                    .conversation_indices
                    .iter()
                    .copied()
                    .filter(|idx| allowed.contains(idx))
                    .collect()
            } else {
                group.conversation_indices.clone()
            };

            // Hide empty groups when filtering.
            if query_active && visible_children.is_empty() {
                continue;
            }

            rows.push(Row::Header { group_idx: gi });

            let expanded = query_active
                || group
                    .canonical_path
                    .as_ref()
                    .is_some_and(|p| self.expanded_groups.contains(p))
                || group.canonical_path.is_none() && query_active;

            if expanded {
                for conv_idx in visible_children {
                    rows.push(Row::Conversation {
                        group_idx: gi,
                        conv_idx,
                    });
                }
            }
        }

        self.rows = rows;

        // Clamp selection.
        self.selected_row = match self.selected_row {
            Some(s) if s < self.rows.len() => Some(s),
            _ if !self.rows.is_empty() => Some(0),
            _ => None,
        };
    }

    /// Toggle expand/collapse on the currently selected group header.
    fn toggle_selected_group(&mut self) {
        let Some(s) = self.selected_row else { return };
        let group_idx = match self.rows.get(s) {
            Some(Row::Header { group_idx }) => *group_idx,
            Some(Row::Conversation { group_idx, .. }) => *group_idx,
            _ => return,
        };
        let Some(path) = self.groups.get(group_idx).and_then(|g| g.canonical_path.clone()) else {
            return;
        };
        if self.expanded_groups.contains(&path) {
            self.expanded_groups.remove(&path);
        } else {
            self.expanded_groups.insert(path);
        }
        self.rebuild_rows();
    }

    fn expand_all_groups(&mut self) {
        for g in &self.groups {
            if let Some(p) = &g.canonical_path {
                self.expanded_groups.insert(p.clone());
            }
        }
        self.rebuild_rows();
    }

    fn collapse_all_groups(&mut self) {
        self.expanded_groups.clear();
        self.rebuild_rows();
    }

    /// Jump selection to the Nth visible group header (1-based).
    fn jump_to_group(&mut self, n: usize) {
        if n == 0 {
            return;
        }
        let mut count = 0usize;
        for (i, row) in self.rows.iter().enumerate() {
            if matches!(row, Row::Header { .. }) {
                count += 1;
                if count == n {
                    self.selected_row = Some(i);
                    self.on_selection_changed();
                    return;
                }
            }
        }
    }

    /// Move row selection up (Grouped mode).
    fn select_row_prev(&mut self) {
        if let Some(s) = self.selected_row
            && s > 0
        {
            self.selected_row = Some(s - 1);
            self.on_selection_changed();
        }
    }

    /// Move row selection down (Grouped mode).
    fn select_row_next(&mut self) {
        if let Some(s) = self.selected_row
            && s + 1 < self.rows.len()
        {
            self.selected_row = Some(s + 1);
            self.on_selection_changed();
        }
    }

    fn select_row_first(&mut self) {
        if !self.rows.is_empty() {
            self.selected_row = Some(0);
            self.on_selection_changed();
        }
    }

    fn select_row_last(&mut self) {
        if !self.rows.is_empty() {
            self.selected_row = Some(self.rows.len() - 1);
            self.on_selection_changed();
        }
    }

    fn select_row_page_up(&mut self) {
        if let Some(s) = self.selected_row {
            self.selected_row = Some(s.saturating_sub(10));
            self.on_selection_changed();
        }
    }

    fn select_row_page_down(&mut self) {
        if let Some(s) = self.selected_row {
            let new = (s + 10).min(self.rows.len().saturating_sub(1));
            self.selected_row = Some(new);
            self.on_selection_changed();
        }
    }

    fn select_row_half_page_up(&mut self, viewport_height: usize) {
        if let Some(s) = self.selected_row {
            self.selected_row = Some(s.saturating_sub(viewport_height / 2));
            self.on_selection_changed();
        }
    }

    fn select_row_half_page_down(&mut self, viewport_height: usize) {
        if let Some(s) = self.selected_row {
            let new = (s + viewport_height / 2).min(self.rows.len().saturating_sub(1));
            self.selected_row = Some(new);
            self.on_selection_changed();
        }
    }

    /// Conversation index for the currently selected row.
    /// If the selected row is a group header, falls through to the group's
    /// most recent conversation so resume/delete/select still target something useful.
    fn selected_row_conv_idx(&self) -> Option<usize> {
        match self.selected_row.and_then(|s| self.rows.get(s)) {
            Some(Row::Conversation { conv_idx, .. }) => Some(*conv_idx),
            Some(Row::Header { group_idx }) => self
                .groups
                .get(*group_idx)
                .and_then(|g| g.conversation_indices.first().copied()),
            None => None,
        }
    }

    /// Toggle between Flat and Grouped list views.
    fn toggle_view_mode(&mut self) {
        self.view_mode = match self.view_mode {
            ListViewMode::Flat => ListViewMode::Grouped,
            ListViewMode::Grouped => ListViewMode::Flat,
        };
        if self.view_mode == ListViewMode::Grouped {
            self.rebuild_groups();
        }
    }

    /// Move selection up
    fn select_prev(&mut self) {
        if let Some(selected) = self.selected
            && selected > 0
        {
            self.selected = Some(selected - 1);
        }
    }

    /// Move selection down
    fn select_next(&mut self) {
        if let Some(selected) = self.selected
            && selected + 1 < self.filtered.len()
        {
            self.selected = Some(selected + 1);
        }
    }

    /// Move selection to first item
    fn select_first(&mut self) {
        if !self.filtered.is_empty() {
            self.selected = Some(0);
        }
    }

    /// Move selection to last item
    fn select_last(&mut self) {
        if !self.filtered.is_empty() {
            self.selected = Some(self.filtered.len() - 1);
        }
    }

    /// Move selection up by a page
    fn select_page_up(&mut self) {
        if let Some(selected) = self.selected {
            self.selected = Some(selected.saturating_sub(10));
        }
    }

    /// Move selection down by a page
    fn select_page_down(&mut self) {
        if let Some(selected) = self.selected {
            let new_selected = (selected + 10).min(self.filtered.len().saturating_sub(1));
            self.selected = Some(new_selected);
        }
    }

    /// Move selection up by half a page (vim-style Ctrl-U)
    fn select_half_page_up(&mut self, viewport_height: usize) {
        if let Some(selected) = self.selected {
            let half_page = viewport_height / 2;
            self.selected = Some(selected.saturating_sub(half_page));
        }
    }

    /// Move selection down by half a page (vim-style Ctrl-D)
    fn select_half_page_down(&mut self, viewport_height: usize) {
        if let Some(selected) = self.selected {
            let half_page = viewport_height / 2;
            let new_selected = (selected + half_page).min(self.filtered.len().saturating_sub(1));
            self.selected = Some(new_selected);
        }
    }

    /// Get the currently selected conversation path
    fn get_selected_path(&self) -> Option<PathBuf> {
        if self.view_mode == ListViewMode::Grouped {
            return self
                .selected_row_conv_idx()
                .map(|idx| self.conversations[idx].path.clone());
        }
        self.selected
            .and_then(|sel| self.filtered.get(sel))
            .map(|&idx| self.conversations[idx].path.clone())
    }

    // Getters for UI access
    pub fn filtered(&self) -> &[usize] {
        &self.filtered
    }

    pub fn conversations(&self) -> &[Conversation] {
        &self.conversations
    }

    pub fn searchable(&self) -> &[SearchableConversation] {
        &self.searchable
    }

    pub fn selected(&self) -> Option<usize> {
        self.selected
    }

    pub fn query(&self) -> &str {
        &self.query
    }

    pub fn query_words(&self) -> &[String] {
        &self.query_words
    }

    pub fn use_relative_time(&self) -> bool {
        self.use_relative_time
    }

    pub fn dialog_mode(&self) -> &DialogMode {
        &self.dialog_mode
    }

    pub fn app_mode(&self) -> &AppMode {
        &self.app_mode
    }

    pub fn status_message(&self) -> Option<&(String, std::time::Instant)> {
        self.status_message.as_ref()
    }

    pub fn cursor_pos(&self) -> usize {
        self.cursor_pos
    }

    pub fn is_single_file_mode(&self) -> bool {
        self.single_file_mode
    }

    pub fn view_mode(&self) -> ListViewMode {
        self.view_mode
    }

    pub fn groups(&self) -> &[ProjectGroup] {
        &self.groups
    }

    pub fn rows(&self) -> &[Row] {
        &self.rows
    }

    pub fn selected_row(&self) -> Option<usize> {
        self.selected_row
    }

    pub fn is_selected_row_header(&self) -> bool {
        matches!(
            self.selected_row.and_then(|s| self.rows.get(s)),
            Some(Row::Header { .. })
        )
    }

    pub fn preview_visible(&self) -> bool {
        self.preview_visible
    }

    pub fn pane_focus(&self) -> PaneFocus {
        self.pane_focus
    }

    pub fn preview_scroll(&self) -> usize {
        self.preview_scroll
    }

    /// Read the cached preview lines (built earlier by `ensure_preview`).
    pub fn preview_lines(&self) -> &[RenderedLine] {
        self.preview_cache
            .as_ref()
            .map(|(_, l)| l.as_slice())
            .unwrap_or(&[])
    }

    pub fn toggle_preview_pane(&mut self) {
        self.preview_visible = !self.preview_visible;
        if !self.preview_visible {
            self.pane_focus = PaneFocus::List;
        }
    }

    pub fn focus_preview(&mut self) {
        if self.preview_visible {
            self.pane_focus = PaneFocus::Preview;
        }
    }

    pub fn focus_list(&mut self) {
        self.pane_focus = PaneFocus::List;
    }

    /// Build/get cached preview lines for the currently selected row.
    pub fn ensure_preview(
        &mut self,
        width: usize,
        providers: &[Box<dyn Provider>],
    ) -> &[RenderedLine] {
        use crate::tui::preview;
        use crate::tui::viewer::RenderOptions;

        let Some(s) = self.selected_row else {
            self.preview_cache = None;
            return &[];
        };
        let Some(row) = self.rows.get(s).cloned() else {
            self.preview_cache = None;
            return &[];
        };

        match row {
            Row::Header { group_idx } => {
                if let Some(group) = self.groups.get(group_idx) {
                    let lines = preview::build_group_preview(group, &self.conversations, width);
                    let key = PreviewKey {
                        path: PathBuf::from(format!("__group__{}", group_idx)),
                        width,
                        tool_display: self.tool_display,
                        show_thinking: self.show_thinking,
                    };
                    self.preview_cache = Some((key, lines));
                } else {
                    self.preview_cache = None;
                }
            }
            Row::Conversation { conv_idx, .. } => {
                let Some(conv) = self.conversations.get(conv_idx).cloned() else {
                    self.preview_cache = None;
                    return &[];
                };
                let key = PreviewKey {
                    path: conv.path.clone(),
                    width,
                    tool_display: self.tool_display,
                    show_thinking: self.show_thinking,
                };
                let needs_rebuild = self
                    .preview_cache
                    .as_ref()
                    .is_none_or(|(k, _)| k != &key);
                if needs_rebuild {
                    let (assistant_label, assistant_color, assistant_dim_color) =
                        match conv.provider {
                            ProviderKind::Claude => {
                                ("Claude".to_string(), (218, 119, 86), (170, 93, 67))
                            }
                            ProviderKind::Cursor => {
                                ("Cursor IDE".to_string(), (180, 130, 230), (140, 100, 180))
                            }
                            ProviderKind::CursorAgent => {
                                ("Cursor CLI".to_string(), (94, 184, 255), (72, 140, 194))
                            }
                        };
                    let options = RenderOptions {
                        tool_display: self.tool_display,
                        show_thinking: self.show_thinking,
                        show_timing: false,
                        content_width: width,
                        assistant_label,
                        assistant_color,
                        assistant_dim_color,
                    };
                    match preview::build_preview(&conv, providers, &options) {
                        Ok(lines) => self.preview_cache = Some((key, lines)),
                        Err(_) => self.preview_cache = None,
                    }
                }
            }
        }

        self.preview_cache
            .as_ref()
            .map(|(_, l)| l.as_slice())
            .unwrap_or(&[])
    }

    /// Reset preview scroll and invalidate when the user moves selection.
    fn on_selection_changed(&mut self) {
        self.preview_scroll = 0;
        // cache is keyed on path; will rebuild on next ensure_preview if path differs
    }

    /// Drop the preview cache so the next `ensure_preview` re-reads from disk.
    /// Used after a resumed session writes new messages to the same file.
    pub fn invalidate_preview(&mut self) {
        self.preview_cache = None;
    }

    /// Maximum valid scroll offset given current preview content + viewport.
    fn preview_max_scroll(&self) -> usize {
        let total = self.preview_lines().len();
        let viewport = self.preview_viewport_height.get();
        total.saturating_sub(viewport)
    }

    /// Cache the preview viewport height observed at render time so scroll
    /// handlers (and `G`) can clamp without a layout reference.
    pub fn set_preview_viewport_height(&self, height: usize) {
        self.preview_viewport_height.set(height);
    }

    fn preview_scroll_down(&mut self, amount: usize) {
        let max = self.preview_max_scroll();
        self.preview_scroll = self.preview_scroll.saturating_add(amount).min(max);
    }

    fn preview_scroll_up(&mut self, amount: usize) {
        self.preview_scroll = self.preview_scroll.saturating_sub(amount);
    }

    fn preview_scroll_to_bottom(&mut self) {
        self.preview_scroll = self.preview_max_scroll();
    }


    pub fn is_group_expanded(&self, group_idx: usize) -> bool {
        let Some(group) = self.groups.get(group_idx) else {
            return false;
        };
        match &group.canonical_path {
            Some(p) => self.expanded_groups.contains(p) || !self.query.is_empty(),
            None => !self.query.is_empty(),
        }
    }

    /// Move cursor left by one character
    fn cursor_left(&mut self) {
        if self.cursor_pos > 0 {
            self.cursor_pos -= 1;
        }
    }

    /// Move cursor right by one character
    fn cursor_right(&mut self) {
        let len = self.query.chars().count();
        if self.cursor_pos < len {
            self.cursor_pos += 1;
        }
    }

    /// Delete the word before the cursor (Ctrl+W behavior).
    /// Returns true if the query was modified.
    fn delete_word_backwards(&mut self) -> bool {
        let chars: Vec<char> = self.query.chars().collect();
        let cursor = self.cursor_pos.min(chars.len());
        if cursor == 0 {
            return false;
        }

        let mut new_pos = cursor;

        // First, consume any separators to the left of cursor
        while new_pos > 0 && search::is_word_separator(chars[new_pos - 1]) {
            new_pos -= 1;
        }

        // Then, consume non-separators (the actual word)
        while new_pos > 0 && !search::is_word_separator(chars[new_pos - 1]) {
            new_pos -= 1;
        }

        if new_pos == cursor {
            return false;
        }

        // Convert char indices to byte indices for safe string manipulation
        let start_byte = self
            .query
            .char_indices()
            .nth(new_pos)
            .map(|(i, _)| i)
            .unwrap_or(0);

        let end_byte = self
            .query
            .char_indices()
            .nth(cursor)
            .map(|(i, _)| i)
            .unwrap_or(self.query.len());

        self.query.replace_range(start_byte..end_byte, "");
        self.cursor_pos = new_pos;
        true
    }

    /// Remove the currently selected conversation from the UI list.
    /// This should only be called after the file has been successfully deleted from disk.
    /// Handles index management for conversations, searchable, and filtered vectors.
    pub fn remove_selected_from_list(&mut self) {
        let Some(selected) = self.selected else {
            return;
        };
        let Some(&conv_idx) = self.filtered.get(selected) else {
            return;
        };

        // Remove from conversations
        self.conversations.remove(conv_idx);

        // Remove from searchable and update indices
        // Note: searchable is not ordered by index due to parallel collection,
        // so we can't use positional removal - must find by index value
        self.searchable.retain_mut(|s| {
            if s.index == conv_idx {
                false // Remove this entry
            } else {
                if s.index > conv_idx {
                    s.index -= 1; // Adjust index for removed item
                }
                true
            }
        });

        // Update filtered: remove the deleted index and decrement all indices > conv_idx
        self.filtered.retain(|&idx| idx != conv_idx);
        for idx in &mut self.filtered {
            if *idx > conv_idx {
                *idx -= 1;
            }
        }

        // Update selection: stay at same position if possible, or move to last item
        if self.filtered.is_empty() {
            self.selected = None;
        } else if selected >= self.filtered.len() {
            self.selected = Some(self.filtered.len() - 1);
        }
        // else: selected stays the same (now pointing to next item)

        if self.view_mode == ListViewMode::Grouped {
            self.rebuild_groups();
        }
    }

    /// Handle a key event during confirmation mode
    fn handle_confirm_key(&mut self, code: KeyCode) -> Option<Action> {
        match code {
            KeyCode::Char('y') | KeyCode::Char('Y') => {
                self.dialog_mode = DialogMode::None;
                self.get_selected_path().map(Action::Delete)
            }
            KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
                self.dialog_mode = DialogMode::None;
                None
            }
            _ => None,
        }
    }

    /// Handle a key event during export/yank menu mode
    fn handle_menu_key(
        &mut self,
        code: KeyCode,
        providers: &[Box<dyn Provider>],
    ) -> Option<Action> {
        let (selected, is_yank) = match &mut self.dialog_mode {
            DialogMode::ExportMenu { selected } => (selected, false),
            DialogMode::YankMenu { selected } => (selected, true),
            _ => return None,
        };

        match code {
            // Navigate up
            KeyCode::Up | KeyCode::Char('k') => {
                *selected = selected.saturating_sub(1);
                None
            }
            // Navigate down
            KeyCode::Down | KeyCode::Char('j') => {
                *selected = (*selected + 1).min(EXPORT_OPTIONS.len() - 1);
                None
            }
            // Number keys for direct selection
            KeyCode::Char('1') => {
                self.perform_export_with_providers(0, is_yank, providers);
                self.dialog_mode = DialogMode::None;
                None
            }
            KeyCode::Char('2') => {
                self.perform_export_with_providers(1, is_yank, providers);
                self.dialog_mode = DialogMode::None;
                None
            }
            KeyCode::Char('3') => {
                self.perform_export_with_providers(2, is_yank, providers);
                self.dialog_mode = DialogMode::None;
                None
            }
            KeyCode::Char('4') => {
                self.perform_export_with_providers(3, is_yank, providers);
                self.dialog_mode = DialogMode::None;
                None
            }
            // Enter to select current option
            KeyCode::Enter => {
                let sel = *selected;
                self.perform_export_with_providers(sel, is_yank, providers);
                self.dialog_mode = DialogMode::None;
                None
            }
            // Escape to cancel
            KeyCode::Esc => {
                self.dialog_mode = DialogMode::None;
                None
            }
            _ => None,
        }
    }

    /// Handle a key event during help overlay mode
    fn handle_help_key(&mut self, code: KeyCode) -> Option<Action> {
        match code {
            KeyCode::Char('?') | KeyCode::Char('q') | KeyCode::Esc => {
                self.dialog_mode = DialogMode::None;
                None
            }
            _ => None,
        }
    }

    /// Perform export or yank operation with provider support
    fn perform_export_with_providers(
        &mut self,
        option: usize,
        to_clipboard: bool,
        providers: &[Box<dyn Provider>],
    ) {
        let path = match &self.app_mode {
            AppMode::View(state) => state.conversation_path.clone(),
            _ => return,
        };

        let format = match crate::tui::export::ExportFormat::from_index(option) {
            Some(f) => f,
            None => return,
        };

        // Try provider-based export for non-JSONL formats
        let conv = self.conversations.iter().find(|c| c.path == path);

        let assistant_label = match conv.map(|c| &c.provider) {
            Some(ProviderKind::Cursor) => "Cursor",
            Some(ProviderKind::CursorAgent) => "Cursor Agent",
            _ => "Claude",
        }
        .to_string();

        let export_options = match &self.app_mode {
            AppMode::View(state) => crate::tui::export::ExportOptions {
                show_tools: state.tool_display.is_visible(),
                show_thinking: state.show_thinking,
                assistant_label,
            },
            _ => return,
        };
        let provider = conv.and_then(|c| providers.iter().find(|p| p.kind() == c.provider));

        let result = if let (Some(provider), Some(conv)) = (provider, conv) {
            // Use provider to read entries for export
            match provider.read_entries(conv) {
                Ok(entries) => {
                    let content = crate::tui::export::generate_content_from_entries(
                        &entries,
                        format,
                        export_options,
                    );
                    if to_clipboard {
                        match arboard::Clipboard::new().and_then(|mut cb| cb.set_text(&content)) {
                            Ok(()) => crate::tui::export::ExportResult {
                                message: "Copied to clipboard".to_string(),
                            },
                            Err(e) => crate::tui::export::ExportResult {
                                message: format!("Clipboard error: {}", e),
                            },
                        }
                    } else {
                        let timestamp = chrono::Local::now().format("%Y-%m-%d-%H%M%S");
                        let ext = format.extension();
                        let filename = format!("conversation-{}.{}", timestamp, ext);
                        match std::fs::write(&filename, &content) {
                            Ok(_) => crate::tui::export::ExportResult {
                                message: format!("Exported to {}", filename),
                            },
                            Err(e) => crate::tui::export::ExportResult {
                                message: format!("Failed to write: {}", e),
                            },
                        }
                    }
                }
                Err(e) => crate::tui::export::ExportResult {
                    message: format!("Failed to read: {}", e),
                },
            }
        } else {
            // Fallback to file-based export
            if to_clipboard {
                crate::tui::export::export_to_clipboard(&path, format, export_options)
            } else {
                crate::tui::export::export_to_file(&path, format, export_options)
            }
        };

        self.status_message = Some((result.message, std::time::Instant::now()));
    }

    /// Handle a key event, returns Some(Action) if the app should exit
    /// viewport_height is the visible content area height for view mode scrolling
    pub fn handle_key(
        &mut self,
        code: KeyCode,
        modifiers: KeyModifiers,
        viewport_height: usize,
        providers: &[Box<dyn Provider>],
    ) -> Option<Action> {
        // Handle dialogs first
        match self.dialog_mode {
            DialogMode::ConfirmDelete => return self.handle_confirm_key(code),
            DialogMode::ExportMenu { .. } | DialogMode::YankMenu { .. } => {
                return self.handle_menu_key(code, providers);
            }
            DialogMode::Help => return self.handle_help_key(code),
            DialogMode::None => {}
        }

        // Delegate based on app mode
        match &self.app_mode {
            AppMode::View(_) => self.handle_view_key(code, modifiers, viewport_height, providers),
            AppMode::List => self.handle_list_key(code, modifiers, viewport_height),
        }
    }

    /// Handle key events in view mode
    fn handle_view_key(
        &mut self,
        code: KeyCode,
        modifiers: KeyModifiers,
        viewport_height: usize,
        providers: &[Box<dyn Provider>],
    ) -> Option<Action> {
        // First check if we're in search typing mode
        if let AppMode::View(ref state) = self.app_mode
            && state.search_mode == ViewSearchMode::Typing
        {
            return self.handle_search_typing_key(code);
        }

        let state = match &mut self.app_mode {
            AppMode::View(s) => s,
            _ => return None,
        };

        let max_scroll = state.total_lines.saturating_sub(viewport_height);

        match code {
            // Exit view mode (or clear search if active)
            KeyCode::Esc => {
                // If search is active, clear it first before exiting view
                if let AppMode::View(ref mut state) = self.app_mode
                    && state.search_mode == ViewSearchMode::Active
                {
                    state.search_mode = ViewSearchMode::Off;
                    state.search_matches.clear();
                    state.search_query.clear();
                    return None;
                }
                // In single file mode, Esc quits the app
                if self.single_file_mode {
                    return Some(Action::Quit);
                }
                self.app_mode = AppMode::List;
                None
            }

            KeyCode::Char('q') => {
                // In single file mode, q quits the app
                if self.single_file_mode {
                    return Some(Action::Quit);
                }
                self.app_mode = AppMode::List;
                None
            }

            // Scroll down one line
            KeyCode::Down | KeyCode::Char('j') => {
                state.scroll_offset = (state.scroll_offset + 1).min(max_scroll);
                None
            }

            // Scroll up one line
            KeyCode::Up | KeyCode::Char('k') => {
                state.scroll_offset = state.scroll_offset.saturating_sub(1);
                None
            }

            // Scroll down half page
            KeyCode::Char('d') if !modifiers.contains(KeyModifiers::CONTROL) => {
                let half_page = viewport_height / 2;
                state.scroll_offset = (state.scroll_offset + half_page).min(max_scroll);
                None
            }

            // Scroll up half page
            KeyCode::Char('u') if !modifiers.contains(KeyModifiers::CONTROL) => {
                let half_page = viewport_height / 2;
                state.scroll_offset = state.scroll_offset.saturating_sub(half_page);
                None
            }

            // Page down
            KeyCode::PageDown => {
                state.scroll_offset = (state.scroll_offset + viewport_height).min(max_scroll);
                None
            }

            // Page up
            KeyCode::PageUp => {
                state.scroll_offset = state.scroll_offset.saturating_sub(viewport_height);
                None
            }

            // Jump to top
            KeyCode::Char('g') | KeyCode::Home => {
                state.scroll_offset = 0;
                None
            }

            // Jump to bottom
            KeyCode::Char('G') | KeyCode::End => {
                state.scroll_offset = max_scroll;
                None
            }

            // Start search
            KeyCode::Char('/') => {
                self.start_view_search();
                None
            }

            // Next match
            KeyCode::Char('n') if !modifiers.contains(KeyModifiers::CONTROL) => {
                if let AppMode::View(ref state) = self.app_mode
                    && state.search_mode == ViewSearchMode::Active
                {
                    self.next_search_match(viewport_height);
                }
                None
            }

            // Previous match
            KeyCode::Char('N') => {
                if let AppMode::View(ref state) = self.app_mode
                    && state.search_mode == ViewSearchMode::Active
                {
                    self.prev_search_match(viewport_height);
                }
                None
            }

            // Toggle tools
            KeyCode::Char('t') => {
                self.toggle_view_tools(viewport_height, providers);
                None
            }

            // Toggle thinking
            KeyCode::Char('T') => {
                self.toggle_view_thinking(viewport_height, providers);
                None
            }

            // Toggle timing (timestamps + durations)
            KeyCode::Char('i') => {
                self.toggle_view_timing(viewport_height, providers);
                None
            }

            // Show path
            KeyCode::Char('p') => {
                if let AppMode::View(ref state) = self.app_mode {
                    self.status_message = Some((
                        state.conversation_path.display().to_string(),
                        std::time::Instant::now(),
                    ));
                }
                None
            }

            // Copy path to clipboard
            KeyCode::Char('Y') => {
                if let AppMode::View(ref state) = self.app_mode {
                    let path_str = state.conversation_path.display().to_string();
                    match arboard::Clipboard::new().and_then(|mut cb| cb.set_text(&path_str)) {
                        Ok(()) => {
                            self.status_message = Some((
                                "Path copied to clipboard".to_string(),
                                std::time::Instant::now(),
                            ));
                        }
                        Err(e) => {
                            self.status_message = Some((
                                format!("Clipboard error: {}", e),
                                std::time::Instant::now(),
                            ));
                        }
                    }
                }
                None
            }

            // Copy session ID to clipboard
            KeyCode::Char('I') => {
                if let AppMode::View(ref state) = self.app_mode
                    && let Some(id) = state.conversation_path.file_stem().and_then(|s| s.to_str())
                {
                    match arboard::Clipboard::new().and_then(|mut cb| cb.set_text(id)) {
                        Ok(()) => {
                            self.status_message = Some((
                                "Session ID copied to clipboard".to_string(),
                                std::time::Instant::now(),
                            ));
                        }
                        Err(e) => {
                            self.status_message = Some((
                                format!("Clipboard error: {}", e),
                                std::time::Instant::now(),
                            ));
                        }
                    }
                }
                None
            }

            // Open export menu (save to file)
            KeyCode::Char('e') => {
                self.dialog_mode = DialogMode::ExportMenu { selected: 0 };
                None
            }

            // Open yank menu (copy to clipboard)
            KeyCode::Char('y') => {
                self.dialog_mode = DialogMode::YankMenu { selected: 0 };
                None
            }

            // Open help overlay
            KeyCode::Char('?') => {
                self.dialog_mode = DialogMode::Help;
                None
            }

            // Ctrl+D - half page down (vim-style, same as 'd')
            KeyCode::Char('d') if modifiers.contains(KeyModifiers::CONTROL) => {
                let half_page = viewport_height / 2;
                state.scroll_offset = (state.scroll_offset + half_page).min(max_scroll);
                None
            }

            // Ctrl+U - half page up (vim-style, same as 'u')
            KeyCode::Char('u') if modifiers.contains(KeyModifiers::CONTROL) => {
                let half_page = viewport_height / 2;
                state.scroll_offset = state.scroll_offset.saturating_sub(half_page);
                None
            }

            // Ctrl+X - delete (disabled in single file mode for security)
            KeyCode::Char('x') if modifiers.contains(KeyModifiers::CONTROL) => {
                if !self.single_file_mode {
                    self.dialog_mode = DialogMode::ConfirmDelete;
                }
                None
            }

            // Ctrl+R - resume (disabled in single file mode)
            KeyCode::Char('r') if modifiers.contains(KeyModifiers::CONTROL) => {
                if self.single_file_mode {
                    None
                } else {
                    self.get_selected_path().map(Action::Resume)
                }
            }

            // Ctrl+C - quit the app
            KeyCode::Char('c') if modifiers.contains(KeyModifiers::CONTROL) => Some(Action::Quit),

            _ => None,
        }
    }

    /// Handle key events while typing a search query
    fn handle_search_typing_key(&mut self, code: KeyCode) -> Option<Action> {
        match code {
            KeyCode::Char(c) => {
                if let AppMode::View(ref mut state) = self.app_mode {
                    state.search_query.push(c);
                }
                self.update_search_results();
                None
            }
            KeyCode::Backspace => {
                if let AppMode::View(ref mut state) = self.app_mode {
                    state.search_query.pop();
                }
                self.update_search_results();
                None
            }
            KeyCode::Enter => {
                if let AppMode::View(ref mut state) = self.app_mode {
                    if !state.search_matches.is_empty() {
                        state.search_mode = ViewSearchMode::Active;
                    } else {
                        state.search_mode = ViewSearchMode::Off;
                    }
                }
                None
            }
            KeyCode::Esc => {
                if let AppMode::View(ref mut state) = self.app_mode {
                    state.search_mode = ViewSearchMode::Off;
                    state.search_query.clear();
                    state.search_matches.clear();
                }
                None
            }
            _ => None,
        }
    }

    /// Handle key events in list mode
    fn handle_list_key(
        &mut self,
        code: KeyCode,
        modifiers: KeyModifiers,
        viewport_height: usize,
    ) -> Option<Action> {
        // During loading, allow navigation and typing but not Enter selection
        if self.is_loading() {
            return match code {
                KeyCode::Esc => Some(Action::Quit),
                KeyCode::Char('c') if modifiers.contains(KeyModifiers::CONTROL) => {
                    Some(Action::Quit)
                }
                KeyCode::Left => {
                    self.cursor_left();
                    None
                }
                KeyCode::Right => {
                    self.cursor_right();
                    None
                }
                KeyCode::Up => {
                    self.select_prev();
                    None
                }
                KeyCode::Down => {
                    self.select_next();
                    None
                }
                KeyCode::Char('n') if modifiers.contains(KeyModifiers::CONTROL) => {
                    self.select_next();
                    None
                }
                KeyCode::Char('p') if modifiers.contains(KeyModifiers::CONTROL) => {
                    self.select_prev();
                    None
                }
                KeyCode::PageUp => {
                    self.select_page_up();
                    None
                }
                KeyCode::PageDown => {
                    self.select_page_down();
                    None
                }
                KeyCode::Char('w') if modifiers.contains(KeyModifiers::CONTROL) => {
                    if self.delete_word_backwards() {
                        self.refresh_query_words();
                    }
                    None
                }
                // Open help overlay
                KeyCode::Char('?') => {
                    self.dialog_mode = DialogMode::Help;
                    None
                }
                // Allow typing during loading - query is buffered for when loading finishes
                KeyCode::Char(c) => {
                    // Insert at cursor position
                    let byte_pos = self
                        .query
                        .char_indices()
                        .nth(self.cursor_pos)
                        .map(|(i, _)| i)
                        .unwrap_or(self.query.len());
                    self.query.insert(byte_pos, c);
                    self.cursor_pos += 1;
                    // Refresh query words cache even during loading so UI highlighting stays in sync
                    self.refresh_query_words();
                    None
                }
                KeyCode::Backspace => {
                    if self.cursor_pos > 0
                        && let Some((byte_pos, _)) =
                            self.query.char_indices().nth(self.cursor_pos - 1)
                    {
                        self.query.remove(byte_pos);
                        self.cursor_pos -= 1;
                        // Refresh query words cache even during loading so UI highlighting stays in sync
                        self.refresh_query_words();
                    }
                    None
                }
                KeyCode::Delete => {
                    let len = self.query.chars().count();
                    if self.cursor_pos < len
                        && let Some((byte_pos, _)) = self.query.char_indices().nth(self.cursor_pos)
                    {
                        self.query.remove(byte_pos);
                        // Refresh query words cache even during loading so UI highlighting stays in sync
                        self.refresh_query_words();
                    }
                    None
                }
                _ => None,
            };
        }

        // Grouped mode has its own selection vector and a few extra keys.
        if self.view_mode == ListViewMode::Grouped {
            // Preview-pane focused: scroll keys go to preview, not the list.
            if self.pane_focus == PaneFocus::Preview {
                match code {
                    KeyCode::Esc | KeyCode::Left | KeyCode::Char('h') => {
                        self.focus_list();
                        return None;
                    }
                    KeyCode::Char('j') | KeyCode::Down => {
                        self.preview_scroll_down(1);
                        return None;
                    }
                    KeyCode::Char('k') | KeyCode::Up => {
                        self.preview_scroll_up(1);
                        return None;
                    }
                    KeyCode::Char('d') if modifiers.contains(KeyModifiers::CONTROL) => {
                        let pv = self.preview_viewport_height.get().max(1);
                        self.preview_scroll_down(pv / 2);
                        return None;
                    }
                    KeyCode::Char('u') if modifiers.contains(KeyModifiers::CONTROL) => {
                        let pv = self.preview_viewport_height.get().max(1);
                        self.preview_scroll_up(pv / 2);
                        return None;
                    }
                    KeyCode::PageDown => {
                        let pv = self.preview_viewport_height.get().max(1);
                        self.preview_scroll_down(pv);
                        return None;
                    }
                    KeyCode::PageUp => {
                        let pv = self.preview_viewport_height.get().max(1);
                        self.preview_scroll_up(pv);
                        return None;
                    }
                    KeyCode::Char('g') => {
                        self.preview_scroll = 0;
                        return None;
                    }
                    KeyCode::Char('G') if !modifiers.contains(KeyModifiers::CONTROL) => {
                        self.preview_scroll_to_bottom();
                        return None;
                    }
                    KeyCode::Char('q') | KeyCode::Char('c')
                        if modifiers.contains(KeyModifiers::CONTROL) =>
                    {
                        return Some(Action::Quit);
                    }
                    // Conversation actions still work while preview is focused —
                    // the selected row in the list is the natural target.
                    KeyCode::Char('r') if modifiers.contains(KeyModifiers::CONTROL) => {
                        return self.get_selected_path().map(Action::Resume);
                    }
                    KeyCode::Char('o') if modifiers.contains(KeyModifiers::CONTROL) => {
                        return self.get_selected_path().map(Action::Select);
                    }
                    KeyCode::Char('x') if modifiers.contains(KeyModifiers::CONTROL) => {
                        if self.get_selected_path().is_some() {
                            self.dialog_mode = DialogMode::ConfirmDelete;
                        }
                        return None;
                    }
                    KeyCode::Enter => {
                        return self.get_selected_path().map(Action::Select);
                    }
                    _ => return None,
                }
            }

            // Group-mode hotkeys handled before generic typing so they aren't shadowed.
            match code {
                KeyCode::Tab => {
                    self.toggle_selected_group();
                    return None;
                }
                KeyCode::BackTab => {
                    self.collapse_all_groups();
                    return None;
                }
                KeyCode::Char('*') => {
                    self.expand_all_groups();
                    return None;
                }
                KeyCode::Char('G') if !modifiers.contains(KeyModifiers::CONTROL) => {
                    self.toggle_view_mode();
                    return None;
                }
                KeyCode::Char('P') => {
                    self.toggle_preview_pane();
                    return None;
                }
                KeyCode::Right if self.preview_visible && self.query.is_empty() => {
                    self.focus_preview();
                    return None;
                }
                KeyCode::Char(c)
                    if c.is_ascii_digit()
                        && c != '0'
                        && self.query.is_empty()
                        && !modifiers.contains(KeyModifiers::CONTROL) =>
                {
                    self.jump_to_group(c.to_digit(10).unwrap_or(0) as usize);
                    return None;
                }
                _ => {}
            }
        }

        // Normal handling when ready
        match code {
            KeyCode::Esc => Some(Action::Quit),
            // Enter now triggers view mode entry (handled in run loop)
            KeyCode::Enter => None,
            KeyCode::Left => {
                self.cursor_left();
                None
            }
            KeyCode::Right => {
                self.cursor_right();
                None
            }
            KeyCode::Up => {
                if self.view_mode == ListViewMode::Grouped {
                    self.select_row_prev();
                } else {
                    self.select_prev();
                }
                None
            }
            KeyCode::Down => {
                if self.view_mode == ListViewMode::Grouped {
                    self.select_row_next();
                } else {
                    self.select_next();
                }
                None
            }
            KeyCode::Home => {
                if self.view_mode == ListViewMode::Grouped {
                    self.select_row_first();
                } else {
                    self.select_first();
                }
                None
            }
            KeyCode::End => {
                if self.view_mode == ListViewMode::Grouped {
                    self.select_row_last();
                } else {
                    self.select_last();
                }
                None
            }
            KeyCode::PageUp => {
                if self.view_mode == ListViewMode::Grouped {
                    self.select_row_page_up();
                } else {
                    self.select_page_up();
                }
                None
            }
            KeyCode::PageDown => {
                if self.view_mode == ListViewMode::Grouped {
                    self.select_row_page_down();
                } else {
                    self.select_page_down();
                }
                None
            }
            KeyCode::Char('c') if modifiers.contains(KeyModifiers::CONTROL) => Some(Action::Quit),
            KeyCode::Char('n') if modifiers.contains(KeyModifiers::CONTROL) => {
                if self.view_mode == ListViewMode::Grouped {
                    self.select_row_next();
                } else {
                    self.select_next();
                }
                None
            }
            KeyCode::Char('p') if modifiers.contains(KeyModifiers::CONTROL) => {
                if self.view_mode == ListViewMode::Grouped {
                    self.select_row_prev();
                } else {
                    self.select_prev();
                }
                None
            }
            // Ctrl+D - half page down (vim-style)
            KeyCode::Char('d') if modifiers.contains(KeyModifiers::CONTROL) => {
                if self.view_mode == ListViewMode::Grouped {
                    self.select_row_half_page_down(viewport_height);
                } else {
                    self.select_half_page_down(viewport_height);
                }
                None
            }
            // Ctrl+U - half page up (vim-style)
            KeyCode::Char('u') if modifiers.contains(KeyModifiers::CONTROL) => {
                if self.view_mode == ListViewMode::Grouped {
                    self.select_row_half_page_up(viewport_height);
                } else {
                    self.select_half_page_up(viewport_height);
                }
                None
            }
            // Ctrl+X - delete conversation
            KeyCode::Char('x') if modifiers.contains(KeyModifiers::CONTROL) => {
                if self.get_selected_path().is_some() {
                    self.dialog_mode = DialogMode::ConfirmDelete;
                }
                None
            }
            KeyCode::Char('r') if modifiers.contains(KeyModifiers::CONTROL) => {
                self.get_selected_path().map(Action::Resume)
            }
            // Ctrl+O - select and exit (for scripting, --show-path)
            KeyCode::Char('o') if modifiers.contains(KeyModifiers::CONTROL) => {
                self.get_selected_path().map(Action::Select)
            }
            KeyCode::Char('w') if modifiers.contains(KeyModifiers::CONTROL) => {
                if self.delete_word_backwards() {
                    self.update_filter();
                }
                None
            }
            // Open help overlay
            KeyCode::Char('?') => {
                self.dialog_mode = DialogMode::Help;
                None
            }
            KeyCode::Char(c) => {
                // Insert at cursor position
                let byte_pos = self
                    .query
                    .char_indices()
                    .nth(self.cursor_pos)
                    .map(|(i, _)| i)
                    .unwrap_or(self.query.len());
                self.query.insert(byte_pos, c);
                self.cursor_pos += 1;
                self.update_filter();
                None
            }
            KeyCode::Backspace => {
                let mut changed = false;
                if self.cursor_pos > 0
                    && let Some((byte_pos, _)) = self.query.char_indices().nth(self.cursor_pos - 1)
                {
                    self.query.remove(byte_pos);
                    self.cursor_pos -= 1;
                    changed = true;
                }
                if changed {
                    self.update_filter();
                }
                None
            }
            KeyCode::Delete => {
                let mut changed = false;
                let len = self.query.chars().count();
                if self.cursor_pos < len
                    && let Some((byte_pos, _)) = self.query.char_indices().nth(self.cursor_pos)
                {
                    self.query.remove(byte_pos);
                    changed = true;
                }
                if changed {
                    self.update_filter();
                }
                None
            }
            _ => None,
        }
    }

    /// Enter view mode for the currently selected conversation
    pub fn enter_view_mode(&mut self, content_width: usize, providers: &[Box<dyn Provider>]) {
        use crate::tui::viewer::{RenderOptions, render_entries};

        let conv_idx = if self.view_mode == ListViewMode::Grouped {
            match self.selected_row_conv_idx() {
                Some(idx) => idx,
                None => return,
            }
        } else {
            let Some(selected) = self.selected else {
                return;
            };
            let Some(&idx) = self.filtered.get(selected) else {
                return;
            };
            idx
        };
        let conv = &self.conversations[conv_idx];
        let path = conv.path.clone();

        let (assistant_label, assistant_color, assistant_dim_color) =
            provider_theme(&conv.provider);

        let options = RenderOptions {
            tool_display: self.tool_display,
            show_thinking: self.show_thinking,
            show_timing: self.show_timing,
            content_width,
            assistant_label,
            assistant_color,
            assistant_dim_color,
        };

        // Find the right provider and read entries
        let provider = providers.iter().find(|p| p.kind() == conv.provider);
        let entries_result = match provider {
            Some(p) => p.read_entries(conv),
            None => {
                // Fallback: read from file path (Claude provider behavior)
                crate::tui::viewer::read_log_entries(&path).map_err(AppError::Io)
            }
        };

        match entries_result {
            Ok(entries) => {
                let rendered_lines = render_entries(&entries, &options);
                let total_lines = rendered_lines.len();
                self.app_mode = AppMode::View(ViewState {
                    conversation_path: path,
                    scroll_offset: 0,
                    rendered_lines,
                    total_lines,
                    tool_display: self.tool_display,
                    show_thinking: self.show_thinking,
                    show_timing: self.show_timing,
                    content_width,
                    search_mode: ViewSearchMode::Off,
                    search_query: String::new(),
                    search_matches: Vec::new(),
                    current_match: 0,
                    cached_entries: entries,
                });
            }
            Err(e) => {
                self.status_message =
                    Some((format!("Failed to open: {}", e), std::time::Instant::now()));
            }
        }
    }

    /// Exit view mode and return to list
    pub fn exit_view_mode(&mut self) {
        self.app_mode = AppMode::List;
    }

    /// Start search mode in view
    fn start_view_search(&mut self) {
        if let AppMode::View(ref mut state) = self.app_mode {
            state.search_mode = ViewSearchMode::Typing;
            state.search_query.clear();
            state.search_matches.clear();
            state.current_match = 0;
        }
    }

    /// Update search results based on current query
    fn update_search_results(&mut self) {
        if let AppMode::View(ref mut state) = self.app_mode {
            let query_lower = state.search_query.to_lowercase();
            if query_lower.is_empty() {
                state.search_matches.clear();
                return;
            }

            state.search_matches = state
                .rendered_lines
                .iter()
                .enumerate()
                .filter(|(_, line)| line_matches_query(line, &query_lower))
                .map(|(i, _)| i)
                .collect();

            // Jump to first match if any
            if !state.search_matches.is_empty() {
                state.current_match = 0;
                state.scroll_offset = state.search_matches[0];
            }
        }
    }

    /// Go to next search match
    fn next_search_match(&mut self, viewport_height: usize) {
        if let AppMode::View(ref mut state) = self.app_mode {
            if state.search_matches.is_empty() {
                return;
            }
            state.current_match = (state.current_match + 1) % state.search_matches.len();
            let match_line = state.search_matches[state.current_match];
            // Scroll to show match in viewport
            if match_line < state.scroll_offset
                || match_line >= state.scroll_offset + viewport_height
            {
                state.scroll_offset = match_line;
            }
        }
    }

    /// Go to previous search match
    fn prev_search_match(&mut self, viewport_height: usize) {
        if let AppMode::View(ref mut state) = self.app_mode {
            if state.search_matches.is_empty() {
                return;
            }
            state.current_match = if state.current_match == 0 {
                state.search_matches.len() - 1
            } else {
                state.current_match - 1
            };
            let match_line = state.search_matches[state.current_match];
            if match_line < state.scroll_offset
                || match_line >= state.scroll_offset + viewport_height
            {
                state.scroll_offset = match_line;
            }
        }
    }

    /// Cycle tool display mode in view mode
    fn toggle_view_tools(&mut self, viewport_height: usize, providers: &[Box<dyn Provider>]) {
        if let AppMode::View(ref mut state) = self.app_mode {
            state.tool_display = state.tool_display.next();
            self.tool_display = state.tool_display; // Persist at app level
            self.re_render_view(viewport_height, providers);
        }
    }

    /// Toggle thinking visibility in view mode
    fn toggle_view_thinking(&mut self, viewport_height: usize, providers: &[Box<dyn Provider>]) {
        if let AppMode::View(ref mut state) = self.app_mode {
            state.show_thinking = !state.show_thinking;
            self.show_thinking = state.show_thinking; // Persist at app level
            self.re_render_view(viewport_height, providers);
        }
    }

    /// Toggle timing visibility in view mode (timestamps + durations)
    fn toggle_view_timing(&mut self, viewport_height: usize, providers: &[Box<dyn Provider>]) {
        if let AppMode::View(ref mut state) = self.app_mode {
            state.show_timing = !state.show_timing;
            self.show_timing = state.show_timing; // Persist at app level
            self.re_render_view(viewport_height, providers);
        }
    }

    /// Re-render the view with current toggle settings
    fn re_render_view(&mut self, viewport_height: usize, providers: &[Box<dyn Provider>]) {
        use crate::tui::viewer::{RenderOptions, render_conversation, render_entries};

        if let AppMode::View(ref mut state) = self.app_mode {
            let conv = self
                .conversations
                .iter()
                .find(|c| c.path == state.conversation_path);

            let provider_kind = conv.map(|c| &c.provider).unwrap_or(&ProviderKind::Claude);
            let (assistant_label, assistant_color, assistant_dim_color) =
                provider_theme(provider_kind);

            let options = RenderOptions {
                tool_display: state.tool_display,
                show_thinking: state.show_thinking,
                show_timing: state.show_timing,
                content_width: state.content_width,
                assistant_label,
                assistant_color,
                assistant_dim_color,
            };

            // Use cached entries if available, otherwise read from disk
            let lines_result = if !state.cached_entries.is_empty() {
                Ok(render_entries(&state.cached_entries, &options))
            } else if let Some(conv) = conv {
                let provider = providers.iter().find(|p| p.kind() == conv.provider);
                match provider {
                    Some(p) => p.read_entries(conv).map(|entries| {
                        let lines = render_entries(&entries, &options);
                        state.cached_entries = entries;
                        lines
                    }),
                    None => render_conversation(&state.conversation_path, &options)
                        .map_err(AppError::Io),
                }
            } else {
                render_conversation(&state.conversation_path, &options).map_err(AppError::Io)
            };

            if let Ok(lines) = lines_result {
                let old_scroll = state.scroll_offset;
                state.total_lines = lines.len();
                state.rendered_lines = lines;

                // Clamp scroll offset to new content bounds
                let max_scroll = state.total_lines.saturating_sub(viewport_height);
                state.scroll_offset = old_scroll.min(max_scroll);

                // Recompute search matches for new content
                if state.search_mode == ViewSearchMode::Active && !state.search_query.is_empty() {
                    let query_lower = state.search_query.to_lowercase();
                    state.search_matches = state
                        .rendered_lines
                        .iter()
                        .enumerate()
                        .filter(|(_, line)| line_matches_query(line, &query_lower))
                        .map(|(i, _)| i)
                        .collect();

                    // Clamp current_match to valid range
                    if state.search_matches.is_empty() {
                        state.current_match = 0;
                    } else {
                        state.current_match =
                            state.current_match.min(state.search_matches.len() - 1);
                    }
                }
            }
        }
    }

    /// Check if view needs re-render due to width change
    pub fn check_view_resize(
        &mut self,
        new_content_width: usize,
        viewport_height: usize,
        providers: &[Box<dyn Provider>],
    ) {
        if let AppMode::View(ref mut state) = self.app_mode
            && state.content_width != new_content_width
        {
            state.content_width = new_content_width;
            self.re_render_view(viewport_height, providers);
        }
    }
}

/// RAII guard to ensure terminal is restored on exit
struct TerminalGuard {
    terminal: Terminal<CrosstermBackend<Stdout>>,
}

/// Check if a rendered line matches the search query by concatenating all span texts.
/// This allows multi-word queries to match across span boundaries.
pub fn line_matches_query(line: &RenderedLine, query_lower: &str) -> bool {
    let full_text: String = line.spans.iter().map(|(text, _)| text.as_str()).collect();
    full_text.to_lowercase().contains(query_lower)
}

impl TerminalGuard {
    fn new() -> Result<Self> {
        terminal::enable_raw_mode().map_err(|e| AppError::Io(io::Error::other(e)))?;

        let mut stdout = io::stdout();
        if let Err(e) = crossterm::execute!(stdout, EnterAlternateScreen) {
            let _ = terminal::disable_raw_mode();
            return Err(AppError::Io(io::Error::other(e)));
        }

        let backend = CrosstermBackend::new(stdout);
        let terminal = match Terminal::new(backend) {
            Ok(t) => t,
            Err(e) => {
                let _ = terminal::disable_raw_mode();
                let _ = crossterm::execute!(io::stdout(), LeaveAlternateScreen);
                return Err(AppError::Io(io::Error::other(e)));
            }
        };

        Ok(Self { terminal })
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = terminal::disable_raw_mode();
        let _ = crossterm::execute!(self.terminal.backend_mut(), LeaveAlternateScreen);
    }
}

/// Hand the terminal back to a child process (e.g. `claude --resume`) by leaving the
/// alternate screen, restoring the cursor, and disabling raw mode. The TUI remains
/// alive in the parent process; pair with `resume_terminal` once the child exits.
fn suspend_terminal(terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> Result<()> {
    terminal::disable_raw_mode().map_err(|e| AppError::Io(io::Error::other(e)))?;
    crossterm::execute!(terminal.backend_mut(), LeaveAlternateScreen, Show)
        .map_err(|e| AppError::Io(io::Error::other(e)))?;
    Ok(())
}

/// Re-acquire the terminal after a foregrounded child has returned: re-enter the
/// alternate screen, hide the cursor, re-enable raw mode, and force a full repaint.
fn resume_terminal(terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> Result<()> {
    terminal::enable_raw_mode().map_err(|e| AppError::Io(io::Error::other(e)))?;
    crossterm::execute!(terminal.backend_mut(), EnterAlternateScreen, Hide)
        .map_err(|e| AppError::Io(io::Error::other(e)))?;
    terminal
        .clear()
        .map_err(|e| AppError::Io(io::Error::other(e)))?;
    Ok(())
}

/// Drain any keystrokes the child process left in the input buffer so they don't
/// accidentally fire dashboard shortcuts on return.
fn drain_pending_events() {
    while let Ok(true) = event::poll(Duration::from_millis(0)) {
        if event::read().is_err() {
            break;
        }
    }
}

/// Run the appropriate provider's resume command for `path`, suspending and restoring
/// the TUI around the call. Errors are logged but never propagated — the dashboard
/// always comes back so the user is never stranded on the shell.
fn handle_resume_action(
    app: &mut App,
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    path: &std::path::Path,
    providers: &[Box<dyn Provider>],
    default_args: &[String],
) {
    let _ = debug_log::log_selected_path(path);

    let conv = app
        .conversations()
        .iter()
        .find(|c| c.path == path)
        .cloned();
    let Some(conv) = conv else {
        let _ = debug_log::log_debug(&format!(
            "Resume requested for unknown conversation: {}",
            path.display()
        ));
        return;
    };
    let Some(provider) = providers.iter().find(|p| p.kind() == conv.provider) else {
        let _ = debug_log::log_debug(&format!(
            "Resume requested but no provider registered for {:?}",
            conv.provider
        ));
        return;
    };

    if let Err(err) = suspend_terminal(terminal) {
        let _ = debug_log::log_debug(&format!("Failed to suspend terminal for resume: {}", err));
        return;
    }

    let resume_result = provider.resume(&conv, default_args);

    if let Err(err) = resume_terminal(terminal) {
        let _ = debug_log::log_debug(&format!("Failed to resume terminal after resume: {}", err));
    }
    drain_pending_events();

    if let Err(err) = resume_result {
        let _ = debug_log::log_debug(&format!("Resume command failed: {}", err));
    }

    // The resumed session almost always appends new messages to the same file,
    // so drop any cached preview lines and let the next frame rebuild from disk.
    app.invalidate_preview();
}

/// Name column width for ledger-style display
const NAME_WIDTH: usize = 9;

/// Run the TUI and return the selected conversation path or None if cancelled
pub fn run(
    conversations: Vec<Conversation>,
    use_relative_time: bool,
    tool_display: ToolDisplayMode,
    show_thinking: bool,
    show_deleted_projects: bool,
    providers: &[Box<dyn Provider>],
    default_args: &[String],
) -> Result<Action> {
    // Set up panic hook to restore terminal
    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        let _ = terminal::disable_raw_mode();
        let _ = crossterm::execute!(io::stdout(), LeaveAlternateScreen);
        original_hook(panic_info);
    }));

    let mut guard = TerminalGuard::new()?;
    let mut app = App::new(
        conversations,
        use_relative_time,
        tool_display,
        show_thinking,
        show_deleted_projects,
    );

    loop {
        let frame_area = guard.terminal.get_frame().area();
        let viewport_height = frame_area.height.saturating_sub(3) as usize; // Subtract header/status
        let content_width = (frame_area.width as usize).saturating_sub(NAME_WIDTH + 3);

        // Check for resize in view mode
        app.check_view_resize(content_width, viewport_height, providers);

        // Pre-build preview lines so the renderer (which only takes &App) can read them.
        if matches!(app.app_mode(), AppMode::List)
            && app.view_mode() == ListViewMode::Grouped
            && app.preview_visible()
            && (frame_area.width as usize) >= 100
        {
            let preview_width = (frame_area.width as usize) / 2;
            app.ensure_preview(preview_width.saturating_sub(4), providers);
        }

        guard.terminal.draw(|frame| ui::render(frame, &app))?;

        if let Event::Key(key) = event::read().map_err(|e| AppError::Io(io::Error::other(e)))? {
            // Only handle key press events (not release)
            if key.kind == KeyEventKind::Press {
                // Check for Enter in list mode - enter view mode (but not during dialogs)
                if matches!(app.app_mode(), AppMode::List)
                    && *app.dialog_mode() == DialogMode::None
                    && key.code == KeyCode::Enter
                    && !app.is_loading()
                {
                    if app.view_mode() == ListViewMode::Grouped {
                        if app.is_selected_row_header() {
                            app.toggle_selected_group();
                            continue;
                        }
                        if app.selected_row().is_some() {
                            app.enter_view_mode(content_width, providers);
                            continue;
                        }
                    } else if app.selected().is_some() {
                        app.enter_view_mode(content_width, providers);
                        continue;
                    }
                }

                if let Some(action) =
                    app.handle_key(key.code, key.modifiers, viewport_height, providers)
                {
                    match action {
                        Action::Delete(ref path) => {
                            // Delete through provider dispatch
                            let conv = app
                                .conversations()
                                .iter()
                                .find(|c| &c.path == path)
                                .cloned();
                            let deleted = if let Some(ref conv) = conv {
                                if let Some(provider) =
                                    providers.iter().find(|p| p.kind() == conv.provider)
                                {
                                    provider.delete(conv).is_ok()
                                } else {
                                    std::fs::remove_file(path).is_ok()
                                }
                            } else {
                                std::fs::remove_file(path).is_ok()
                            };
                            if deleted {
                                app.remove_selected_from_list();
                                app.exit_view_mode();
                            } else {
                                let _ = debug_log::log_debug(&format!(
                                    "Failed to delete {}",
                                    path.display(),
                                ));
                            }
                        }
                        Action::Select(ref path) => {
                            let _ = debug_log::log_selected_path(path);
                            return Ok(action);
                        }
                        Action::Resume(ref path) => {
                            let path = path.clone();
                            handle_resume_action(
                                &mut app,
                                &mut guard.terminal,
                                &path,
                                providers,
                                default_args,
                            );
                        }
                        Action::Quit => return Ok(action),
                    }
                }
            }
        }
    }
}

/// Run the TUI with background loading
/// Returns the action and the final list of conversations
pub fn run_with_loader(
    rx: Receiver<LoaderMessage>,
    use_relative_time: bool,
    tool_display: ToolDisplayMode,
    show_thinking: bool,
    show_deleted_projects: bool,
    providers: &[Box<dyn Provider>],
    default_args: &[String],
) -> Result<(Action, Vec<Conversation>)> {
    // Set up panic hook to restore terminal
    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        let _ = terminal::disable_raw_mode();
        let _ = crossterm::execute!(io::stdout(), LeaveAlternateScreen);
        original_hook(panic_info);
    }));

    let mut guard = TerminalGuard::new()?;
    let mut app = App::new_loading(
        use_relative_time,
        tool_display,
        show_thinking,
        show_deleted_projects,
    );

    loop {
        // Process all pending loader messages (non-blocking)
        loop {
            match rx.try_recv() {
                Ok(LoaderMessage::Fatal(err)) => {
                    // Fatal error - restore terminal and return error
                    drop(guard);
                    return Err(err);
                }
                Ok(LoaderMessage::ProjectError) => {
                    // Logged by loader, continue
                }
                Ok(LoaderMessage::Batch(convs)) => {
                    app.append_conversations(convs);
                }
                Ok(LoaderMessage::Done) => {
                    app.finish_loading();
                    // Check for empty conversations
                    if app.conversations().is_empty() {
                        drop(guard);
                        return Err(AppError::NoHistoryFound("selected scope".to_string()));
                    }
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    // Loader finished unexpectedly
                    if app.is_loading() {
                        app.finish_loading();
                        if app.conversations().is_empty() {
                            drop(guard);
                            return Err(AppError::NoHistoryFound("selected scope".to_string()));
                        }
                    }
                    break;
                }
            }
        }

        let frame_area = guard.terminal.get_frame().area();
        let viewport_height = frame_area.height.saturating_sub(3) as usize;
        let content_width = (frame_area.width as usize).saturating_sub(NAME_WIDTH + 3);

        // Check for resize in view mode
        app.check_view_resize(content_width, viewport_height, providers);

        if matches!(app.app_mode(), AppMode::List)
            && app.view_mode() == ListViewMode::Grouped
            && app.preview_visible()
            && (frame_area.width as usize) >= 100
        {
            let preview_width = (frame_area.width as usize) / 2;
            app.ensure_preview(preview_width.saturating_sub(4), providers);
        }

        // Render current state
        guard.terminal.draw(|frame| ui::render(frame, &app))?;

        // Poll for keyboard input with timeout (allows us to check loader messages)
        if event::poll(Duration::from_millis(50)).map_err(|e| AppError::Io(io::Error::other(e)))?
            && let Event::Key(key) = event::read().map_err(|e| AppError::Io(io::Error::other(e)))?
            && key.kind == KeyEventKind::Press
        {
            // Check for Enter in list mode - enter view mode (but not during dialogs)
            if matches!(app.app_mode(), AppMode::List)
                && *app.dialog_mode() == DialogMode::None
                && key.code == KeyCode::Enter
                && !app.is_loading()
                && app.selected().is_some()
            {
                app.enter_view_mode(content_width, providers);
                continue;
            }

            if let Some(action) =
                app.handle_key(key.code, key.modifiers, viewport_height, providers)
            {
                match action {
                    Action::Delete(ref path) => {
                        // Delete through provider dispatch
                        let conv = app
                            .conversations()
                            .iter()
                            .find(|c| &c.path == path)
                            .cloned();
                        let deleted = if let Some(ref conv) = conv {
                            if let Some(provider) =
                                providers.iter().find(|p| p.kind() == conv.provider)
                            {
                                provider.delete(conv).is_ok()
                            } else {
                                std::fs::remove_file(path).is_ok()
                            }
                        } else {
                            std::fs::remove_file(path).is_ok()
                        };
                        if deleted {
                            app.remove_selected_from_list();
                            app.exit_view_mode();
                        } else {
                            let _ = debug_log::log_debug(&format!(
                                "Failed to delete {}",
                                path.display(),
                            ));
                        }
                    }
                    Action::Resume(ref path) => {
                        let path = path.clone();
                        handle_resume_action(
                            &mut app,
                            &mut guard.terminal,
                            &path,
                            providers,
                            default_args,
                        );
                    }
                    _ => return Ok((action, app.into_conversations())),
                }
            }
        }
    }
}

/// Run the TUI for a single file (direct input mode)
pub fn run_single_file(
    path: PathBuf,
    use_relative_time: bool,
    tool_display: ToolDisplayMode,
    show_thinking: bool,
    providers: &[Box<dyn Provider>],
) -> Result<()> {
    // Set up panic hook to restore terminal
    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        let _ = terminal::disable_raw_mode();
        let _ = crossterm::execute!(io::stdout(), LeaveAlternateScreen);
        original_hook(panic_info);
    }));

    let mut guard = TerminalGuard::new()?;
    let mut app = App::new_single_file(path, use_relative_time, tool_display, show_thinking);

    loop {
        let frame_area = guard.terminal.get_frame().area();
        let viewport_height = frame_area.height.saturating_sub(3) as usize;
        let content_width = (frame_area.width as usize).saturating_sub(NAME_WIDTH + 3);

        // Check for resize in view mode (this triggers initial render too)
        app.check_view_resize(content_width, viewport_height, providers);

        guard.terminal.draw(|frame| ui::render(frame, &app))?;

        if let Event::Key(key) = event::read().map_err(|e| AppError::Io(io::Error::other(e)))?
            && key.kind == KeyEventKind::Press
            && let Some(Action::Quit) =
                app.handle_key(key.code, key.modifiers, viewport_height, providers)
        {
            return Ok(());
        }
    }
}
