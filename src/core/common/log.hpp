// SPDX-License-Identifier: GPL-3.0-or-later
// pylidar — common::log
//
// Process-global logging hook. Default callback is a no-op. The Python entry
// point `pylidar.set_log_callback(callable)` wires this to a Python callable
// (re-acquiring the GIL inside the binding).
//
// Header-only in v0.1: a Meyer's-singleton-style `inline` accessor gives all
// translation units the same callback instance. When the core gets enough .cpp
// files to warrant a static library we can move the storage out-of-line.
//
// Thread-safety caveat: the callback pointer is *not* atomic. Don't change it
// while algorithm code is running. The Python entry point only wires it once
// at startup in normal usage.

#pragma once

#include <functional>
#include <string>
#include <string_view>
#include <utility>

namespace pylidar::common::log {

using Callback = std::function<void(std::string_view)>;

namespace detail {
inline Callback& storage() noexcept {
    static Callback cb;
    return cb;
}
}  // namespace detail

// Replace the global log callback. Pass an empty std::function to reset to the
// default no-op. Returns the previous callback (useful for stacking).
inline Callback set_callback(Callback cb) noexcept {
    Callback prev = std::move(detail::storage());
    detail::storage() = std::move(cb);
    return prev;
}

// Read-only access to the currently registered callback. Cheap.
inline const Callback& get_callback() noexcept {
    return detail::storage();
}

// Emit a log line. No-op if no callback is set.
inline void emit(std::string_view msg) {
    const Callback& cb = get_callback();
    if (cb) {
        cb(msg);
    }
}

}  // namespace pylidar::common::log
