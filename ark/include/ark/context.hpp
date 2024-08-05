// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_CONTEXT_HPP
#define ARK_CONTEXT_HPP

#include <ark/model.hpp>

namespace ark {

enum class ContextType {
    Overwrite,
    Extend,
    Immutable,
};

class Context {
   public:
    ///
    /// Construct an empty context for the given model.
    ///
    /// @param model The model to create the context for.
    ///
    Context(Model& model);

    /// Get the ID of this context.
    int id() const;

    /// Get context value by key.
    /// @param key The key of the context item.
    /// @return The value of the context item. If the key does not exist,
    ///         an empty string is returned.
    std::string get(const std::string& key) const;

    ///
    /// Add an item to the context.
    ///
    /// The given context item is valid for the lifetime of the context
    /// object. @p `value` is assumed to be a JSON string.
    /// If @p `key` is already in use by another valid context item
    /// of either the same or different context object for the same model,
    /// the behavior is determined by the context type @p `type` as follows.
    ///
    /// - `ContextType::Overwrite` (default): The existing value will be
    /// replaced with the new one while the context object is alive.
    /// When the context object is destroyed, the previous value will be
    /// restored.
    ///
    /// - `ContextType::Extend`: The new value will extend the existing
    /// value while the context object is alive. This type is feasible only
    /// when the value represents a JSON object, which is convertible to a
    /// map. If the new JSON object has a key that already exists in the
    /// existing JSON object, the value of the existing key will be
    /// overwritten by the new value. When the context object is destroyed,
    /// the previous value will be restored.
    ///
    /// - `ContextType::Immutable`: The new value will be adopted only when the
    /// key does not exist in the existing context or when the value of the key
    /// is empty. If the key already exists, the new value will be ignored.
    /// When the context object is destroyed, if the key did not exist in the
    /// existing context, the key will be removed.
    /// Otherwise, nothing will be changed.
    ///
    /// @param key The key of the context item.
    /// @param value The value of the context item. The value is assumed to
    /// be a JSON string. An empty JSON string is also allowed.
    /// @param type The context type. Default is `ContextType::Overwrite`.
    ///
    /// @throw `InvalidUsageError` In the following cases:
    ///
    /// - The value cannot be parsed as JSON.
    ///
    /// - The value is not a JSON object when the context type is
    /// `ContextType::Extend`.
    ///
    /// - The context type is unknown.
    ///
    void set(const std::string& key, const std::string& value,
             ContextType type = ContextType::Overwrite);

   protected:
    friend class PlannerContext;

    class Impl;
    std::shared_ptr<Impl> impl_;
};

}  // namespace ark

#endif  // ARK_CONTEXT_HPP
