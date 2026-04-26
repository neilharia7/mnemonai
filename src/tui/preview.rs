//! Right-pane preview rendering: header metadata + tail of the conversation body.
//!
//! Built lazily per selection change. Reuses `viewer::render_entries` so the body
//! preview honors the same tool/thinking toggles as the full viewer.

use crate::error::Result;
use crate::history::Conversation;
use crate::providers::Provider;
use crate::tui::app::{LineStyle, RenderedLine};
use crate::tui::viewer::{self, RenderOptions};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;

/// Lookup the current git branch for a project. Cached per path.
pub fn project_branch(path: &Path) -> Option<String> {
    static CACHE: Mutex<Option<HashMap<PathBuf, Option<String>>>> = Mutex::new(None);

    let mut guard = CACHE.lock().ok()?;
    let map = guard.get_or_insert_with(HashMap::new);
    if let Some(cached) = map.get(path) {
        return cached.clone();
    }

    let result = Command::new("git")
        .args(["-C"])
        .arg(path)
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .output()
        .ok()
        .and_then(|out| {
            if out.status.success() {
                String::from_utf8(out.stdout)
                    .ok()
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty() && s != "HEAD")
            } else {
                None
            }
        });

    map.insert(path.to_path_buf(), result.clone());
    result
}

fn label(text: &str) -> (String, LineStyle) {
    (
        format!("{:<9}", text),
        LineStyle {
            fg: Some((140, 140, 140)),
            ..Default::default()
        },
    )
}

fn value(text: String, color: (u8, u8, u8)) -> (String, LineStyle) {
    (
        text,
        LineStyle {
            fg: Some(color),
            ..Default::default()
        },
    )
}

fn plain(text: String) -> (String, LineStyle) {
    (
        text,
        LineStyle {
            fg: Some((200, 200, 200)),
            ..Default::default()
        },
    )
}

fn header_line(label_text: &str, val: &str, val_color: (u8, u8, u8)) -> RenderedLine {
    RenderedLine {
        spans: vec![label(label_text), value(val.to_string(), val_color)],
    }
}

fn empty_line() -> RenderedLine {
    RenderedLine { spans: Vec::new() }
}

fn rule(width: usize) -> RenderedLine {
    RenderedLine {
        spans: vec![(
            "─".repeat(width),
            LineStyle {
                fg: Some((60, 60, 60)),
                ..Default::default()
            },
        )],
    }
}

fn format_tokens(tokens: u64) -> String {
    if tokens >= 1_000_000 {
        format!("{:.1}M", tokens as f64 / 1_000_000.0)
    } else if tokens >= 1_000 {
        format!("{}k", tokens / 1_000)
    } else {
        tokens.to_string()
    }
}

fn format_duration(minutes: u64) -> String {
    if minutes >= 60 {
        format!("{}h {}m", minutes / 60, minutes % 60)
    } else {
        format!("{}m", minutes)
    }
}

fn truncate_id(id: &str) -> String {
    if id.len() > 13 {
        format!("{}…", &id[..12])
    } else {
        id.to_string()
    }
}

fn short_path(path: &Path) -> String {
    if let Some(home) = home::home_dir()
        && let Ok(rest) = path.strip_prefix(&home)
    {
        return format!("~/{}", rest.display());
    }
    path.display().to_string()
}

/// Status line for the conversation: ●/◐/○/✗ + label.
fn status_for(conv: &Conversation) -> (&'static str, &'static str, (u8, u8, u8)) {
    use chrono::Local;
    if !conv.parse_errors.is_empty() {
        return ("✗", "parse errors", (220, 90, 90));
    }
    if let Some(p) = &conv.project_path
        && !p.exists()
    {
        return ("✗", "project deleted", (220, 90, 90));
    }
    let mins = Local::now()
        .signed_duration_since(conv.timestamp)
        .num_minutes();
    if mins <= 10 {
        ("●", "active", (120, 200, 130))
    } else if mins <= 120 {
        ("◐", "recent", (220, 190, 100))
    } else {
        ("○", "idle", (120, 120, 120))
    }
}

/// Build the preview content for a conversation. Returns a vector of styled lines
/// suitable for paragraph rendering in the right pane.
pub fn build_preview(
    conv: &Conversation,
    providers: &[Box<dyn Provider>],
    options: &RenderOptions,
) -> Result<Vec<RenderedLine>> {
    let mut lines: Vec<RenderedLine> = Vec::new();
    let (glyph, status_label, status_color) = status_for(conv);

    let project_label = conv.project_name.as_deref().unwrap_or("(unknown)");

    // Title row: project · status
    let provider_color = match conv.provider {
        crate::history::ProviderKind::Claude => (218, 119, 86),
        crate::history::ProviderKind::Cursor => (180, 130, 230),
        crate::history::ProviderKind::CursorAgent => (94, 184, 255),
    };
    lines.push(RenderedLine {
        spans: vec![
            (
                project_label.to_string(),
                LineStyle {
                    fg: Some((78, 201, 176)),
                    bold: true,
                    ..Default::default()
                },
            ),
            ("  ".to_string(), LineStyle::default()),
            (
                glyph.to_string(),
                LineStyle {
                    fg: Some(status_color),
                    ..Default::default()
                },
            ),
            (" ".to_string(), LineStyle::default()),
            (
                status_label.to_string(),
                LineStyle {
                    fg: Some(status_color),
                    ..Default::default()
                },
            ),
            ("  ".to_string(), LineStyle::default()),
            (
                provider_label(&conv.provider).to_string(),
                LineStyle {
                    fg: Some(provider_color),
                    ..Default::default()
                },
            ),
        ],
    });
    lines.push(empty_line());

    // Path + branch + session/model/tokens/etc.
    if let Some(p) = &conv.project_path {
        lines.push(header_line("path:", &short_path(p), (200, 200, 200)));
        if let Some(branch) = project_branch(p) {
            lines.push(header_line("branch:", &branch, (180, 140, 200)));
        }
    }
    if let Some(cwd) = &conv.cwd
        && Some(cwd) != conv.project_path.as_ref()
    {
        lines.push(header_line("cwd:", &short_path(cwd), (200, 200, 200)));
    }

    lines.push(header_line("session:", &truncate_id(&conv.id), (140, 140, 140)));

    if let Some(model) = &conv.model {
        lines.push(header_line("model:", &format_model_short(model), (180, 140, 200)));
    }

    if conv.total_tokens > 0 {
        lines.push(header_line(
            "tokens:",
            &format_tokens(conv.total_tokens),
            (140, 140, 140),
        ));
    }

    let msgs_dur = if let Some(m) = conv.duration_minutes {
        format!("{} msgs · {}", conv.message_count, format_duration(m))
    } else {
        format!("{} msgs", conv.message_count)
    };
    lines.push(header_line("messages:", &msgs_dur, (140, 140, 140)));

    lines.push(header_line(
        "updated:",
        &conv.timestamp.format("%Y-%m-%d %H:%M").to_string(),
        (140, 140, 140),
    ));

    if let Some(summary) = &conv.summary
        && !summary.is_empty()
    {
        lines.push(empty_line());
        lines.push(RenderedLine {
            spans: vec![plain(summary.clone())],
        });
    }

    lines.push(empty_line());
    lines.push(rule(options.content_width.max(20)));
    lines.push(empty_line());

    // Body: full transcript, rendered with the user's current toggles.
    let provider = providers.iter().find(|p| p.kind() == conv.provider);
    let entries_result = match provider {
        Some(p) => p.read_entries(conv),
        None => viewer::read_log_entries(&conv.path).map_err(crate::error::AppError::Io),
    };

    match entries_result {
        Ok(entries) => {
            let body_lines = viewer::render_entries(&entries, options);
            lines.extend(body_lines);
        }
        Err(_) => {
            lines.push(RenderedLine {
                spans: vec![value(
                    "(failed to read conversation file)".to_string(),
                    (220, 90, 90),
                )],
            });
        }
    }

    Ok(lines)
}

fn provider_label(kind: &crate::history::ProviderKind) -> &'static str {
    match kind {
        crate::history::ProviderKind::Claude => "Claude",
        crate::history::ProviderKind::Cursor => "Cursor IDE",
        crate::history::ProviderKind::CursorAgent => "Cursor CLI",
    }
}

/// Build a placeholder preview when a group header is selected (no specific conv).
pub fn build_group_preview(
    group: &crate::history::grouping::ProjectGroup,
    convs: &[Conversation],
    width: usize,
) -> Vec<RenderedLine> {
    let mut lines = Vec::new();
    lines.push(RenderedLine {
        spans: vec![(
            group.display_name.clone(),
            LineStyle {
                fg: Some((78, 201, 176)),
                bold: true,
                ..Default::default()
            },
        )],
    });
    lines.push(empty_line());

    if let Some(p) = &group.canonical_path {
        lines.push(header_line("path:", &short_path(p), (200, 200, 200)));
        if let Some(branch) = project_branch(p) {
            lines.push(header_line("branch:", &branch, (180, 140, 200)));
        }
    }
    lines.push(header_line(
        "sessions:",
        &group.conversation_indices.len().to_string(),
        (140, 140, 140),
    ));
    lines.push(header_line(
        "last:",
        &group.last_activity.format("%Y-%m-%d %H:%M").to_string(),
        (140, 140, 140),
    ));

    let mut providers_str = String::new();
    let mut first = true;
    for kind in &group.providers {
        if !first {
            providers_str.push_str(", ");
        }
        providers_str.push_str(provider_label(kind));
        first = false;
    }
    if !providers_str.is_empty() {
        lines.push(header_line("providers:", &providers_str, (140, 140, 140)));
    }

    lines.push(empty_line());
    lines.push(rule(width.max(20)));
    lines.push(empty_line());

    // Top 5 conversations as quick links.
    for (i, &idx) in group.conversation_indices.iter().take(5).enumerate() {
        if let Some(c) = convs.get(idx) {
            let summary = c
                .summary
                .as_deref()
                .filter(|s| !s.is_empty())
                .or(Some(c.preview.trim()))
                .unwrap_or("(empty)");
            let truncated: String = summary.chars().take(width.saturating_sub(8)).collect();
            lines.push(RenderedLine {
                spans: vec![
                    (
                        format!("{:>2}. ", i + 1),
                        LineStyle {
                            fg: Some((100, 100, 100)),
                            ..Default::default()
                        },
                    ),
                    (
                        truncated,
                        LineStyle {
                            fg: Some((200, 200, 200)),
                            ..Default::default()
                        },
                    ),
                ],
            });
        }
    }

    lines
}

fn format_model_short(model: &str) -> String {
    for (prefix, label) in [
        ("claude-opus-4-5-", "opus-4.5"),
        ("claude-opus-4-7-", "opus-4.7"),
        ("claude-sonnet-4-6-", "sonnet-4.6"),
        ("claude-sonnet-4-", "sonnet-4"),
        ("claude-3-5-sonnet-", "sonnet-3.5"),
        ("claude-3-5-haiku-", "haiku-3.5"),
        ("claude-3-opus-", "opus-3"),
        ("claude-3-sonnet-", "sonnet-3"),
        ("claude-3-haiku-", "haiku-3"),
    ] {
        if let Some(rest) = model.strip_prefix(prefix)
            && rest.chars().all(|c| c.is_ascii_digit())
        {
            return label.to_string();
        }
    }
    if model.len() > 20 {
        format!("{}…", &model[..19])
    } else {
        model.to_string()
    }
}
