//! Group conversations by project path for the dashboard view.

use crate::history::{Conversation, ProviderKind, format_short_name_from_path};
use chrono::{DateTime, Local};
use std::collections::HashSet;
use std::path::PathBuf;

/// A bucket of conversations sharing the same project path.
#[derive(Clone, Debug)]
pub struct ProjectGroup {
    /// Stable identity for persistence and selection. `None` = unknown bucket.
    pub canonical_path: Option<PathBuf>,
    /// Human-readable label.
    pub display_name: String,
    /// Indices into the source conversations slice, sorted newest first.
    pub conversation_indices: Vec<usize>,
    /// Most recent conversation timestamp in the group.
    pub last_activity: DateTime<Local>,
    /// Providers present in this group (for status badges).
    pub providers: HashSet<ProviderKind>,
}

/// Stable key for a conversation's project. Falls back to the encoded
/// directory name from the JSONL file path so unrelated unknowns don't all
/// collapse into one bucket.
fn group_key(conv: &Conversation) -> Option<PathBuf> {
    if let Some(p) = &conv.project_path {
        return Some(canonicalize_or_clone(p));
    }
    if let Some(p) = &conv.cwd {
        return Some(canonicalize_or_clone(p));
    }
    None
}

fn canonicalize_or_clone(p: &std::path::Path) -> PathBuf {
    std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf())
}

/// Bucket conversations by project path. Groups are sorted by `last_activity`
/// descending; within each group conversations are ordered newest first.
pub fn group_by_project_path(conversations: &[Conversation]) -> Vec<ProjectGroup> {
    let mut by_key: std::collections::HashMap<Option<PathBuf>, ProjectGroup> =
        std::collections::HashMap::new();

    for (idx, conv) in conversations.iter().enumerate() {
        let key = group_key(conv);
        let entry = by_key
            .entry(key.clone())
            .or_insert_with(|| ProjectGroup {
                canonical_path: key.clone(),
                display_name: display_name_for(conv, key.as_deref()),
                conversation_indices: Vec::new(),
                last_activity: conv.timestamp,
                providers: HashSet::new(),
            });
        entry.conversation_indices.push(idx);
        if conv.timestamp > entry.last_activity {
            entry.last_activity = conv.timestamp;
        }
        entry.providers.insert(conv.provider.clone());
    }

    let mut groups: Vec<ProjectGroup> = by_key.into_values().collect();

    // Sort conversations within each group: newest first.
    for g in &mut groups {
        g.conversation_indices
            .sort_by(|a, b| conversations[*b].timestamp.cmp(&conversations[*a].timestamp));
    }

    // Sort groups by last activity desc; unknown bucket last when tied.
    groups.sort_by(|a, b| {
        b.last_activity
            .cmp(&a.last_activity)
            .then_with(|| match (a.canonical_path.is_some(), b.canonical_path.is_some()) {
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                _ => std::cmp::Ordering::Equal,
            })
    });

    groups
}

fn display_name_for(conv: &Conversation, key: Option<&std::path::Path>) -> String {
    if let Some(path) = key {
        return format_short_name_from_path(path);
    }
    conv.project_name
        .clone()
        .unwrap_or_else(|| "<unknown>".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::ParseError;
    use chrono::TimeZone;

    fn make_conv(
        path: &str,
        project: Option<&str>,
        ts: DateTime<Local>,
        provider: ProviderKind,
    ) -> Conversation {
        Conversation {
            path: PathBuf::from(format!("/tmp/{}.jsonl", path)),
            index: 0,
            provider,
            id: path.to_string(),
            timestamp: ts,
            preview: String::new(),
            full_text: String::new(),
            project_name: project.map(String::from),
            project_path: project.map(PathBuf::from),
            cwd: None,
            message_count: 0,
            parse_errors: Vec::<ParseError>::new(),
            summary: None,
            model: None,
            total_tokens: 0,
            duration_minutes: None,
            search_text_lower: None,
            search_topic_end: None,
        }
    }

    fn ts(secs: i64) -> DateTime<Local> {
        Local.timestamp_opt(secs, 0).unwrap()
    }

    #[test]
    fn buckets_by_project_path() {
        let convs = vec![
            make_conv("a", Some("/code/proj-a"), ts(100), ProviderKind::Claude),
            make_conv("b", Some("/code/proj-b"), ts(200), ProviderKind::Claude),
            make_conv("c", Some("/code/proj-a"), ts(300), ProviderKind::Cursor),
        ];
        let groups = group_by_project_path(&convs);
        assert_eq!(groups.len(), 2);
        // Newest activity first: proj-a (300) before proj-b (200)
        assert_eq!(groups[0].canonical_path.as_deref(), Some(std::path::Path::new("/code/proj-a")));
        assert_eq!(groups[0].conversation_indices, vec![2, 0]);
        assert_eq!(groups[0].last_activity, ts(300));
        assert!(groups[0].providers.contains(&ProviderKind::Claude));
        assert!(groups[0].providers.contains(&ProviderKind::Cursor));

        assert_eq!(groups[1].canonical_path.as_deref(), Some(std::path::Path::new("/code/proj-b")));
        assert_eq!(groups[1].conversation_indices, vec![1]);
    }

    #[test]
    fn unknown_bucket_for_missing_path() {
        let convs = vec![
            make_conv("a", None, ts(100), ProviderKind::Claude),
            make_conv("b", Some("/code/proj"), ts(200), ProviderKind::Claude),
        ];
        let groups = group_by_project_path(&convs);
        assert_eq!(groups.len(), 2);
        // proj has activity 200, unknown has 100 → proj first
        assert!(groups[0].canonical_path.is_some());
        assert!(groups[1].canonical_path.is_none());
        assert_eq!(groups[1].display_name, "<unknown>");
    }

    #[test]
    fn last_activity_is_max_in_group() {
        let convs = vec![
            make_conv("a", Some("/code/proj"), ts(100), ProviderKind::Claude),
            make_conv("b", Some("/code/proj"), ts(500), ProviderKind::Claude),
            make_conv("c", Some("/code/proj"), ts(300), ProviderKind::Claude),
        ];
        let groups = group_by_project_path(&convs);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].last_activity, ts(500));
        // Sorted newest first: b(500), c(300), a(100)
        assert_eq!(groups[0].conversation_indices, vec![1, 2, 0]);
    }
}
