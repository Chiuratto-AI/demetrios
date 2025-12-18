//! I/O Module for Demetrios
//!
//! Provides file I/O operations, process control, and environment access.
//! All I/O operations require the `IO` effect.

// =============================================================================
// Error Types
// =============================================================================

/// Error type for I/O operations
pub enum IoError {
    /// File or directory not found
    NotFound { path: String },
    /// Permission denied
    PermissionDenied { path: String },
    /// Error reading file
    ReadError { path: String, message: String },
    /// Error writing file
    WriteError { path: String, message: String },
    /// Invalid path encoding
    InvalidPath { path: String },
    /// File already exists (when creating exclusively)
    AlreadyExists { path: String },
    /// Directory not empty (when removing)
    DirectoryNotEmpty { path: String },
    /// Operation interrupted
    Interrupted,
    /// Other I/O error
    Other { message: String },
}

impl IoError {
    /// Create a "not found" error
    pub fn not_found(path: String) -> IoError {
        IoError::NotFound { path: path }
    }

    /// Create a "permission denied" error
    pub fn permission_denied(path: String) -> IoError {
        IoError::PermissionDenied { path: path }
    }

    /// Create a "read error"
    pub fn read_error(path: String, message: String) -> IoError {
        IoError::ReadError { path: path, message: message }
    }

    /// Create a "write error"
    pub fn write_error(path: String, message: String) -> IoError {
        IoError::WriteError { path: path, message: message }
    }

    /// Create an "other" error
    pub fn other(message: String) -> IoError {
        IoError::Other { message: message }
    }

    /// Get error message
    pub fn message(self) -> String {
        match self {
            IoError::NotFound { path } => "File not found: " ++ path,
            IoError::PermissionDenied { path } => "Permission denied: " ++ path,
            IoError::ReadError { path, message } => "Read error on " ++ path ++ ": " ++ message,
            IoError::WriteError { path, message } => "Write error on " ++ path ++ ": " ++ message,
            IoError::InvalidPath { path } => "Invalid path: " ++ path,
            IoError::AlreadyExists { path } => "File already exists: " ++ path,
            IoError::DirectoryNotEmpty { path } => "Directory not empty: " ++ path,
            IoError::Interrupted => "Operation interrupted".to_string(),
            IoError::Other { message } => message,
        }
    }
}

impl ToString for IoError {
    fn to_string(self) -> String {
        self.message()
    }
}

impl Debug for IoError {
    fn fmt(self, f: &mut Formatter) -> Result<(), FmtError> {
        f.write_str(self.message())
    }
}

// =============================================================================
// File Operations
// =============================================================================

/// Read entire file contents as a string
///
/// # Arguments
/// * `path` - Path to the file to read
///
/// # Returns
/// * `Ok(String)` - File contents as a string
/// * `Err(IoError)` - Error if file cannot be read
///
/// # Example
/// ```d
/// let content = read_file("config.txt")?;
/// println("File contains: {}", content);
/// ```
pub fn read_file(path: &str) -> Result<String, IoError> with IO {
    // Implementation via FFI to system calls
    // This is a specification - actual implementation in compiler runtime
    extern "C" {
        fn __demetrios_read_file(path: *const u8, path_len: i64, out_ptr: *mut *mut u8, out_len: *mut i64) -> i32;
    }

    var out_ptr: *mut u8 = null_ptr();
    var out_len: i64 = 0;

    let result = unsafe {
        __demetrios_read_file(path.as_ptr(), path.len() as i64, &mut out_ptr, &mut out_len)
    };

    match result {
        0 => {
            // Success - convert raw bytes to String
            let content = unsafe { String::from_raw_parts(out_ptr, out_len as usize) };
            Ok(content)
        },
        1 => Err(IoError::not_found(path.to_string())),
        2 => Err(IoError::permission_denied(path.to_string())),
        _ => Err(IoError::read_error(path.to_string(), "Unknown error".to_string())),
    }
}

/// Write string contents to a file
///
/// Creates the file if it doesn't exist, overwrites if it does.
///
/// # Arguments
/// * `path` - Path to the file to write
/// * `content` - String content to write
///
/// # Returns
/// * `Ok(())` - Success
/// * `Err(IoError)` - Error if file cannot be written
///
/// # Example
/// ```d
/// write_file("output.txt", "Hello, world!")?;
/// ```
pub fn write_file(path: &str, content: &str) -> Result<(), IoError> with IO {
    extern "C" {
        fn __demetrios_write_file(path: *const u8, path_len: i64, content: *const u8, content_len: i64) -> i32;
    }

    let result = unsafe {
        __demetrios_write_file(
            path.as_ptr(), path.len() as i64,
            content.as_ptr(), content.len() as i64
        )
    };

    match result {
        0 => Ok(()),
        1 => Err(IoError::not_found(path.to_string())),
        2 => Err(IoError::permission_denied(path.to_string())),
        _ => Err(IoError::write_error(path.to_string(), "Unknown error".to_string())),
    }
}

/// Append string contents to a file
///
/// Creates the file if it doesn't exist.
///
/// # Arguments
/// * `path` - Path to the file
/// * `content` - String content to append
///
/// # Returns
/// * `Ok(())` - Success
/// * `Err(IoError)` - Error if file cannot be written
pub fn append_file(path: &str, content: &str) -> Result<(), IoError> with IO {
    extern "C" {
        fn __demetrios_append_file(path: *const u8, path_len: i64, content: *const u8, content_len: i64) -> i32;
    }

    let result = unsafe {
        __demetrios_append_file(
            path.as_ptr(), path.len() as i64,
            content.as_ptr(), content.len() as i64
        )
    };

    match result {
        0 => Ok(()),
        1 => Err(IoError::not_found(path.to_string())),
        2 => Err(IoError::permission_denied(path.to_string())),
        _ => Err(IoError::write_error(path.to_string(), "Unknown error".to_string())),
    }
}

/// Check if a file exists
///
/// # Arguments
/// * `path` - Path to check
///
/// # Returns
/// * `true` if file exists, `false` otherwise
pub fn file_exists(path: &str) -> bool with IO {
    extern "C" {
        fn __demetrios_file_exists(path: *const u8, path_len: i64) -> i32;
    }

    let result = unsafe {
        __demetrios_file_exists(path.as_ptr(), path.len() as i64)
    };

    result == 1
}

/// Delete a file
///
/// # Arguments
/// * `path` - Path to the file to delete
///
/// # Returns
/// * `Ok(())` - Success
/// * `Err(IoError)` - Error if file cannot be deleted
pub fn remove_file(path: &str) -> Result<(), IoError> with IO {
    extern "C" {
        fn __demetrios_remove_file(path: *const u8, path_len: i64) -> i32;
    }

    let result = unsafe {
        __demetrios_remove_file(path.as_ptr(), path.len() as i64)
    };

    match result {
        0 => Ok(()),
        1 => Err(IoError::not_found(path.to_string())),
        2 => Err(IoError::permission_denied(path.to_string())),
        _ => Err(IoError::other("Failed to remove file".to_string())),
    }
}

// =============================================================================
// Process Control
// =============================================================================

/// Exit the process with the given exit code
///
/// This function never returns (divergent type `!`).
///
/// # Arguments
/// * `code` - Exit code (0 for success, non-zero for error)
///
/// # Example
/// ```d
/// if error_occurred {
///     exit(1);
/// }
/// exit(0);
/// ```
pub fn exit(code: i32) -> ! with IO {
    extern "C" {
        fn __demetrios_exit(code: i32) -> !;
    }

    unsafe {
        __demetrios_exit(code)
    }
}

// =============================================================================
// Environment
// =============================================================================

/// Module for environment access
pub mod env {
    /// Get command line arguments
    ///
    /// Returns a vector of strings where:
    /// - First element is the program name
    /// - Subsequent elements are command line arguments
    ///
    /// # Example
    /// ```d
    /// // If run as: ./program input.txt output.txt
    /// let args = env::args();
    /// // args = ["./program", "input.txt", "output.txt"]
    /// let input = args[1];
    /// let output = args[2];
    /// ```
    pub fn args() -> Vec<String> with IO {
        extern "C" {
            fn __demetrios_get_argc() -> i32;
            fn __demetrios_get_argv(index: i32, out_ptr: *mut *const u8, out_len: *mut i64) -> i32;
        }

        let argc = unsafe { __demetrios_get_argc() };
        var result: Vec<String> = Vec::with_capacity(argc as usize);

        for i in 0..argc {
            var arg_ptr: *const u8 = null_ptr();
            var arg_len: i64 = 0;

            let status = unsafe {
                __demetrios_get_argv(i, &mut arg_ptr, &mut arg_len)
            };

            if status == 0 {
                let arg = unsafe { String::from_raw_parts(arg_ptr as *mut u8, arg_len as usize) };
                result.push(arg);
            }
        }

        result
    }

    /// Get an environment variable
    ///
    /// # Arguments
    /// * `key` - Environment variable name
    ///
    /// # Returns
    /// * `Some(value)` if the variable exists
    /// * `None` if the variable doesn't exist
    ///
    /// # Example
    /// ```d
    /// match env::var("HOME") {
    ///     Some(home) => println("Home directory: {}", home),
    ///     None => println("HOME not set"),
    /// }
    /// ```
    pub fn var(key: &str) -> Option<String> with IO {
        extern "C" {
            fn __demetrios_get_env(key: *const u8, key_len: i64, out_ptr: *mut *const u8, out_len: *mut i64) -> i32;
        }

        var out_ptr: *const u8 = null_ptr();
        var out_len: i64 = 0;

        let result = unsafe {
            __demetrios_get_env(key.as_ptr(), key.len() as i64, &mut out_ptr, &mut out_len)
        };

        if result == 0 {
            let value = unsafe { String::from_raw_parts(out_ptr as *mut u8, out_len as usize) };
            Some(value)
        } else {
            None
        }
    }

    /// Set an environment variable
    ///
    /// # Arguments
    /// * `key` - Environment variable name
    /// * `value` - Value to set
    pub fn set_var(key: &str, value: &str) with IO {
        extern "C" {
            fn __demetrios_set_env(key: *const u8, key_len: i64, value: *const u8, value_len: i64);
        }

        unsafe {
            __demetrios_set_env(
                key.as_ptr(), key.len() as i64,
                value.as_ptr(), value.len() as i64
            )
        }
    }

    /// Get the current working directory
    ///
    /// # Returns
    /// * `Ok(path)` - Current directory path
    /// * `Err(IoError)` - Error if cannot determine current directory
    pub fn current_dir() -> Result<String, IoError> with IO {
        extern "C" {
            fn __demetrios_current_dir(out_ptr: *mut *const u8, out_len: *mut i64) -> i32;
        }

        var out_ptr: *const u8 = null_ptr();
        var out_len: i64 = 0;

        let result = unsafe {
            __demetrios_current_dir(&mut out_ptr, &mut out_len)
        };

        if result == 0 {
            let path = unsafe { String::from_raw_parts(out_ptr as *mut u8, out_len as usize) };
            Ok(path)
        } else {
            Err(IoError::other("Cannot determine current directory".to_string()))
        }
    }
}

// =============================================================================
// Standard Streams
// =============================================================================

/// Print to standard output
pub fn print(s: &str) with IO {
    extern "C" {
        fn __demetrios_print(s: *const u8, len: i64);
    }

    unsafe {
        __demetrios_print(s.as_ptr(), s.len() as i64)
    }
}

/// Print to standard output with newline
pub fn println(s: &str) with IO {
    print(s);
    print("\n");
}

/// Print to standard error
pub fn eprint(s: &str) with IO {
    extern "C" {
        fn __demetrios_eprint(s: *const u8, len: i64);
    }

    unsafe {
        __demetrios_eprint(s.as_ptr(), s.len() as i64)
    }
}

/// Print to standard error with newline
pub fn eprintln(s: &str) with IO {
    eprint(s);
    eprint("\n");
}

/// Read a line from standard input
pub fn read_line() -> Result<String, IoError> with IO {
    extern "C" {
        fn __demetrios_read_line(out_ptr: *mut *mut u8, out_len: *mut i64) -> i32;
    }

    var out_ptr: *mut u8 = null_ptr();
    var out_len: i64 = 0;

    let result = unsafe {
        __demetrios_read_line(&mut out_ptr, &mut out_len)
    };

    if result == 0 {
        let line = unsafe { String::from_raw_parts(out_ptr, out_len as usize) };
        Ok(line)
    } else {
        Err(IoError::other("Failed to read from stdin".to_string()))
    }
}

// =============================================================================
// Path Utilities
// =============================================================================

/// Module for path manipulation
pub mod path {
    /// Join two path components
    ///
    /// # Example
    /// ```d
    /// let full = path::join("dir", "file.txt");
    /// // full = "dir/file.txt"
    /// ```
    pub fn join(base: &str, component: &str) -> String {
        if base.is_empty() {
            return component.to_string();
        }
        if component.is_empty() {
            return base.to_string();
        }
        if base.ends_with("/") || base.ends_with("\\") {
            return base.to_string() ++ component;
        }
        base.to_string() ++ "/" ++ component
    }

    /// Get the file name from a path
    ///
    /// # Example
    /// ```d
    /// let name = path::file_name("/home/user/file.txt");
    /// // name = Some("file.txt")
    /// ```
    pub fn file_name(p: &str) -> Option<String> {
        let last_slash = p.rfind('/').or_else(|| p.rfind('\\'));
        match last_slash {
            Some(idx) => {
                let name = p[(idx + 1)..];
                if name.is_empty() {
                    None
                } else {
                    Some(name.to_string())
                }
            },
            None => {
                if p.is_empty() {
                    None
                } else {
                    Some(p.to_string())
                }
            },
        }
    }

    /// Get the parent directory from a path
    ///
    /// # Example
    /// ```d
    /// let parent = path::parent("/home/user/file.txt");
    /// // parent = Some("/home/user")
    /// ```
    pub fn parent(p: &str) -> Option<String> {
        let last_slash = p.rfind('/').or_else(|| p.rfind('\\'));
        match last_slash {
            Some(0) => Some("/".to_string()),
            Some(idx) => Some(p[..idx].to_string()),
            None => None,
        }
    }

    /// Get the file extension
    ///
    /// # Example
    /// ```d
    /// let ext = path::extension("file.txt");
    /// // ext = Some("txt")
    /// ```
    pub fn extension(p: &str) -> Option<String> {
        let name = file_name(p)?;
        let last_dot = name.rfind('.')?;
        if last_dot == 0 {
            None
        } else {
            Some(name[(last_dot + 1)..].to_string())
        }
    }
}
