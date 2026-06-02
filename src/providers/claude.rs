use crate::claude::LogEntry;
use crate::conversation_index::delete_conversation;
use crate::error::{AppError, Result};
use crate::history::{self, Conversation, LoaderMessage, ProviderKind};
use crate::tui::viewer;
use std::process::Command;
use std::sync::mpsc::Receiver;

pub struct ClaudeProvider {
    current_dir: Option<std::path::PathBuf>,
    exclude_paths: Vec<String>,
}

impl ClaudeProvider {
    pub fn new(exclude_paths: Vec<String>) -> Self {
        Self {
            current_dir: std::env::current_dir().ok(),
            exclude_paths,
        }
    }
}

impl super::Provider for ClaudeProvider {
    fn kind(&self) -> ProviderKind {
        ProviderKind::Claude
    }

    fn name(&self) -> &str {
        "Claude Code"
    }

    fn detect(&self) -> bool {
        // Claude is always available if ~/.claude/projects exists
        history::get_claude_projects_root()
            .map(|p| p.exists())
            .unwrap_or(false)
    }

    fn load_conversations(
        &self,
        show_last: bool,
        debug: Option<crate::cli::DebugLevel>,
    ) -> Result<Vec<Conversation>> {
        let current_dir = self.current_dir.as_ref().ok_or_else(|| {
            AppError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "Failed to get current directory",
            ))
        })?;
        let projects_dir = history::get_claude_projects_dir(current_dir)?;
        if !projects_dir.exists() {
            return Ok(Vec::new());
        }
        history::load_conversations(&projects_dir, show_last, debug)
    }

    fn load_conversations_streaming(
        &self,
        show_last: bool,
        debug: Option<crate::cli::DebugLevel>,
    ) -> Receiver<LoaderMessage> {
        history::load_all_conversations_streaming(show_last, debug, self.exclude_paths.clone())
    }

    fn read_entries(&self, conversation: &Conversation) -> Result<Vec<LogEntry>> {
        viewer::read_log_entries(&conversation.path).map_err(AppError::Io)
    }

    fn resume(&self, conversation: &Conversation, default_args: &[String]) -> Result<()> {
        let project_dir = match &conversation.project_path {
            Some(path) if path.exists() && path.is_dir() => path,
            Some(path) => {
                return Err(AppError::ClaudeExecutionError(format!(
                    "Project directory no longer exists: {}",
                    path.display()
                )));
            }
            None => {
                return Err(AppError::ClaudeExecutionError(
                    "Cannot determine project directory for this conversation".to_string(),
                ));
            }
        };

        let mut command = Command::new("claude");
        command.args(["--resume", &conversation.id]);
        command.args(default_args);
        command.current_dir(project_dir);

        run_claude_command(command)
    }

    fn delete(&self, conversation: &Conversation) -> Result<()> {
        std::fs::remove_file(&conversation.path).map_err(AppError::Io)?;
        delete_conversation(ProviderKind::Claude, &conversation.path);
        Ok(())
    }
}

fn run_claude_command(mut command: Command) -> Result<()> {
    let status = command
        .status()
        .map_err(|e| AppError::ClaudeExecutionError(e.to_string()))?;

    if treat_status_as_success(&status) {
        Ok(())
    } else {
        Err(AppError::ClaudeExecutionError(format!(
            "claude CLI exited with status {}",
            status
        )))
    }
}

/// Treat clean exits and SIGINT (Ctrl+C) as a successful return so the dashboard
/// resurfaces silently when the user quits the resumed session.
fn treat_status_as_success(status: &std::process::ExitStatus) -> bool {
    if status.success() {
        return true;
    }
    // 130 is the conventional exit code for SIGINT (Ctrl+C); the user quit on purpose.
    matches!(status.code(), Some(130))
}

#[cfg(test)]
mod tests {
    use super::treat_status_as_success;

    #[cfg(unix)]
    #[test]
    fn sigint_is_treated_as_success() {
        use std::os::unix::process::ExitStatusExt;
        let status = std::process::ExitStatus::from_raw(130 << 8);
        assert!(treat_status_as_success(&status));
    }

    #[cfg(unix)]
    #[test]
    fn nonzero_failure_is_not_success() {
        use std::os::unix::process::ExitStatusExt;
        let status = std::process::ExitStatus::from_raw(1 << 8);
        assert!(!treat_status_as_success(&status));
    }
}
