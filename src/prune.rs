//! Bulk deletion of conversations by timestamp.
//!
//! Loads conversations across all providers, filters by a cutoff timestamp,
//! optionally moves the underlying files to a trash directory, then invokes
//! each provider's `delete` to remove the live source and cache row.

use crate::cli::{PruneArgs, ProviderFilter, TimestampSource};
use crate::error::{AppError, Result};
use crate::history::{Conversation, ProviderKind};
use crate::providers::Provider;
use chrono::{DateTime, Local, NaiveDate, TimeZone};
use std::fs;
use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

pub fn run(args: PruneArgs, providers: &[Box<dyn Provider>]) -> Result<()> {
    let cutoff = resolve_cutoff(&args)?;
    let min_age = parse_duration(&args.min_age)
        .ok_or_else(|| AppError::ConfigError(format!("invalid --min-age: {}", args.min_age)))?;
    let provider_filter = args.provider.unwrap_or(ProviderFilter::All);

    let now = Local::now();
    let min_age_threshold = now - chrono::Duration::from_std(min_age).unwrap_or_default();

    let mut victims: Vec<Conversation> = Vec::new();
    for provider in providers {
        if !provider_matches(provider.kind(), provider_filter) {
            continue;
        }

        // show_last=false matches the cache shape used in the default browse path.
        let convs = match provider.load_conversations(false, None) {
            Ok(convs) => convs,
            Err(err) => {
                eprintln!(
                    "warn: failed to load {} conversations: {}",
                    provider_label(provider.kind()),
                    err
                );
                continue;
            }
        };

        for conv in convs {
            if let Some(project) = &args.project {
                let matches = conv
                    .project_path
                    .as_ref()
                    .map(|p| p.to_string_lossy().contains(project.as_str()))
                    .unwrap_or(false)
                    || conv
                        .project_name
                        .as_deref()
                        .map(|n| n.contains(project.as_str()))
                        .unwrap_or(false);
                if !matches {
                    continue;
                }
            }

            if args.exclude_deleted_projects
                && let Some(project_path) = &conv.project_path
                && !project_path.exists()
            {
                continue;
            }

            let (compare_ts, used_mtime) = compare_timestamp(&conv, args.r#use);
            if compare_ts >= cutoff {
                continue;
            }

            // Active-session guard: skip files modified more recently than min-age.
            if conv.timestamp >= min_age_threshold {
                continue;
            }

            let _ = used_mtime; // recorded for display; nothing else to do
            victims.push(conv);
        }
    }

    if victims.is_empty() {
        println!("No conversations to delete (cutoff {}).", cutoff.to_rfc3339());
        return Ok(());
    }

    victims.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));

    print_victim_list(&victims, args.r#use, &cutoff);

    let interactive = io::stdout().is_terminal();
    let dry_run = args.dry_run || (!args.yes && !interactive);

    if dry_run {
        if !args.dry_run && !interactive {
            println!("(no TTY; refusing to delete without --yes — re-run with --yes or --dry-run)");
        }
        return Ok(());
    }

    if !args.yes {
        let total_bytes = victims.iter().filter_map(|c| file_size(&c.path)).sum::<u64>();
        print!(
            "Delete {} conversation(s) ({})? [y/N] ",
            victims.len(),
            format_bytes(total_bytes)
        );
        io::stdout().flush().ok();
        let mut answer = String::new();
        io::stdin().read_line(&mut answer)?;
        if !answer.trim().eq_ignore_ascii_case("y") && !answer.trim().eq_ignore_ascii_case("yes") {
            println!("Aborted.");
            return Err(AppError::SelectionCancelled);
        }
    }

    let trash_root = if args.trash {
        Some(prepare_trash_dir(now)?)
    } else {
        None
    };

    let mut deleted = 0usize;
    let mut failed = 0usize;
    for conv in &victims {
        let provider = match providers.iter().find(|p| p.kind() == conv.provider) {
            Some(p) => p,
            None => {
                eprintln!("error: no provider registered for {:?}", conv.provider);
                failed += 1;
                continue;
            }
        };

        if let Some(root) = &trash_root {
            if let Err(err) = move_to_trash(conv, root) {
                eprintln!(
                    "warn: trash move failed for {}: {} — falling back to delete",
                    conv.path.display(),
                    err
                );
            }
        }

        match provider.delete(conv) {
            Ok(()) => deleted += 1,
            Err(err) => {
                eprintln!("error deleting {}: {}", conv.path.display(), err);
                failed += 1;
            }
        }
    }

    println!(
        "Deleted {}/{} conversation(s){}.",
        deleted,
        victims.len(),
        if failed > 0 {
            format!(" ({} failed)", failed)
        } else {
            String::new()
        }
    );

    if failed == victims.len() {
        return Err(AppError::ClaudeExecutionError(
            "All deletions failed".to_string(),
        ));
    }

    Ok(())
}

fn provider_matches(kind: ProviderKind, filter: ProviderFilter) -> bool {
    match (filter, kind) {
        (ProviderFilter::All, _) => true,
        (ProviderFilter::Claude, ProviderKind::Claude) => true,
        (ProviderFilter::Cursor, ProviderKind::Cursor) => true,
        (ProviderFilter::CursorAgent, ProviderKind::CursorAgent) => true,
        _ => false,
    }
}

fn provider_label(kind: ProviderKind) -> &'static str {
    match kind {
        ProviderKind::Claude => "Claude",
        ProviderKind::Cursor => "Cursor",
        ProviderKind::CursorAgent => "Cursor Agent",
    }
}

fn resolve_cutoff(args: &PruneArgs) -> Result<DateTime<Local>> {
    match (&args.older_than, &args.before) {
        (Some(older_than), None) => {
            let dur = parse_duration(older_than).ok_or_else(|| {
                AppError::ConfigError(format!("invalid --older-than: {}", older_than))
            })?;
            let chrono_dur = chrono::Duration::from_std(dur)
                .map_err(|e| AppError::ConfigError(format!("duration overflow: {}", e)))?;
            Ok(Local::now() - chrono_dur)
        }
        (None, Some(before)) => {
            let date = NaiveDate::parse_from_str(before, "%Y-%m-%d").map_err(|e| {
                AppError::ConfigError(format!("invalid --before date '{}': {}", before, e))
            })?;
            Local
                .from_local_datetime(&date.and_hms_opt(0, 0, 0).unwrap())
                .single()
                .ok_or_else(|| {
                    AppError::ConfigError(format!("ambiguous local time for {}", before))
                })
        }
        (Some(_), Some(_)) => Err(AppError::ConfigError(
            "specify only one of --older-than or --before".to_string(),
        )),
        (None, None) => Err(AppError::ConfigError(
            "must specify --older-than or --before".to_string(),
        )),
    }
}

/// Returns (timestamp_to_compare, used_mtime_fallback).
fn compare_timestamp(conv: &Conversation, source: TimestampSource) -> (DateTime<Local>, bool) {
    match source {
        TimestampSource::Mtime => (conv.timestamp, true),
        TimestampSource::FirstMessage => match conv.first_message_time {
            Some(t) => (t, false),
            None => (conv.timestamp, true),
        },
        TimestampSource::LastMessage => match conv.last_message_time {
            Some(t) => (t, false),
            None => (conv.timestamp, true),
        },
    }
}

fn print_victim_list(
    victims: &[Conversation],
    source: TimestampSource,
    cutoff: &DateTime<Local>,
) {
    println!(
        "Conversations older than {} ({}):",
        cutoff.format("%Y-%m-%d %H:%M"),
        match source {
            TimestampSource::Mtime => "by mtime",
            TimestampSource::FirstMessage => "by first message",
            TimestampSource::LastMessage => "by last message",
        }
    );
    for conv in victims {
        let (ts, used_mtime) = compare_timestamp(conv, source);
        let project = conv.project_name.as_deref().unwrap_or("-");
        let tag = if used_mtime { " (mtime)" } else { "" };
        println!(
            "  {} {:14}  {:3} msg  {}{}  {}",
            ts.format("%Y-%m-%d %H:%M"),
            provider_label(conv.provider.clone()),
            conv.message_count,
            project,
            tag,
            conv.path.display()
        );
    }
    println!("Total: {}", victims.len());
}

fn parse_duration(input: &str) -> Option<Duration> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return None;
    }
    let (num_str, unit) = trimmed.split_at(trimmed.len() - 1);
    let n: u64 = num_str.parse().ok()?;
    let secs = match unit {
        "s" => n,
        "m" => n.checked_mul(60)?,
        "h" => n.checked_mul(3_600)?,
        "d" => n.checked_mul(86_400)?,
        "w" => n.checked_mul(7 * 86_400)?,
        // Months are approximated as 30 days; documented in --help.
        "M" => n.checked_mul(30 * 86_400)?,
        _ => return None,
    };
    Some(Duration::from_secs(secs))
}

fn file_size(path: &Path) -> Option<u64> {
    fs::metadata(path).ok().map(|m| m.len())
}

fn format_bytes(bytes: u64) -> String {
    const KIB: u64 = 1024;
    const MIB: u64 = KIB * 1024;
    const GIB: u64 = MIB * 1024;
    if bytes >= GIB {
        format!("{:.2} GiB", bytes as f64 / GIB as f64)
    } else if bytes >= MIB {
        format!("{:.2} MiB", bytes as f64 / MIB as f64)
    } else if bytes >= KIB {
        format!("{:.2} KiB", bytes as f64 / KIB as f64)
    } else {
        format!("{} B", bytes)
    }
}

fn prepare_trash_dir(now: DateTime<Local>) -> Result<PathBuf> {
    let home = std::env::var("HOME")
        .map_err(|_| AppError::ConfigError("HOME not set".to_string()))?;
    let root = PathBuf::from(home)
        .join(".local")
        .join("state")
        .join("mnemonai")
        .join("trash")
        .join(now.format("%Y%m%dT%H%M%S").to_string());
    fs::create_dir_all(&root)?;
    Ok(root)
}

fn move_to_trash(conv: &Conversation, root: &Path) -> Result<()> {
    if conv.provider == ProviderKind::Cursor {
        // Cursor IDE conversations live in a shared SQLite DB; can't be trashed
        // as a file. Skip silently — the SQL DELETE still runs.
        return Ok(());
    }

    if !conv.path.exists() {
        return Ok(());
    }

    let provider_dir = root.join(provider_label(conv.provider.clone()).replace(' ', "-"));
    fs::create_dir_all(&provider_dir)?;

    let filename = conv
        .path
        .file_name()
        .map(|n| n.to_os_string())
        .unwrap_or_else(|| std::ffi::OsString::from(&conv.id));
    let dest = provider_dir.join(filename);

    // For cursor-agent the conversation may live inside a session directory;
    // try to move the parent dir if it matches the conversation id.
    if conv.provider == ProviderKind::CursorAgent
        && let Some(parent) = conv.path.parent()
        && parent.file_name().and_then(|n| n.to_str()) == Some(conv.id.as_str())
    {
        let dest_dir = provider_dir.join(&conv.id);
        return rename_or_copy(parent, &dest_dir);
    }

    rename_or_copy(&conv.path, &dest)
}

fn rename_or_copy(src: &Path, dest: &Path) -> Result<()> {
    match fs::rename(src, dest) {
        Ok(()) => Ok(()),
        Err(_) => {
            // Cross-device fallback: copy then remove.
            if src.is_dir() {
                copy_dir_recursive(src, dest)?;
                fs::remove_dir_all(src)?;
            } else {
                fs::copy(src, dest)?;
                fs::remove_file(src)?;
            }
            Ok(())
        }
    }
}

fn copy_dir_recursive(src: &Path, dest: &Path) -> Result<()> {
    fs::create_dir_all(dest)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let path = entry.path();
        let target = dest.join(entry.file_name());
        if path.is_dir() {
            copy_dir_recursive(&path, &target)?;
        } else {
            fs::copy(&path, &target)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_duration_units() {
        assert_eq!(parse_duration("30s"), Some(Duration::from_secs(30)));
        assert_eq!(parse_duration("5m"), Some(Duration::from_secs(300)));
        assert_eq!(parse_duration("2h"), Some(Duration::from_secs(7200)));
        assert_eq!(parse_duration("3d"), Some(Duration::from_secs(259200)));
        assert_eq!(parse_duration("1w"), Some(Duration::from_secs(604800)));
        assert_eq!(parse_duration("1M"), Some(Duration::from_secs(2592000)));
        assert_eq!(parse_duration(""), None);
        assert_eq!(parse_duration("xyz"), None);
        assert_eq!(parse_duration("5"), None);
    }

    #[test]
    fn format_bytes_units() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(2048), "2.00 KiB");
        assert_eq!(format_bytes(2 * 1024 * 1024), "2.00 MiB");
    }
}
