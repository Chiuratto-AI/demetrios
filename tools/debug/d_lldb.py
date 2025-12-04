#!/usr/bin/env python3
"""LLDB pretty printers and commands for Demetrios (D).

Load in LLDB with:
    command script import /path/to/d_lldb.py

Or add to your ~/.lldbinit:
    command script import /path/to/demetrios/tools/debug/d_lldb.py
"""

import lldb

# =============================================================================
# Summary Providers
# =============================================================================

def d_string_summary(valobj, internal_dict):

def d_string_summary(valobj, internal_dict):
    """Summary provider for D's String type."""
    try:
        ptr = valobj.GetChildMemberWithName("ptr").GetValueAsUnsigned()
        length = valobj.GetChildMemberWithName("len").GetValueAsUnsigned()
        capacity = valobj.GetChildMemberWithName("cap").GetValueAsUnsigned()

        if ptr == 0:
            return "String(null)"

        # Read string data
        process = valobj.GetProcess()
        error = lldb.SBError()
        data = process.ReadCStringFromMemory(ptr, min(length + 1, 256), error)

        if error.Success():
            if len(data) > length:
                data = data[:length]
            if len(data) > 100:
                data = data[:100] + "..."
            return f'String("{data}", len={length}, cap={capacity})'
        else:
            return f"String(<read error>)"
    except Exception as e:
        return f"String(<error: {e}>)"


def d_vec_summary(valobj, internal_dict):
    """Summary provider for D's Vec<T> type."""
    try:
        length = valobj.GetChildMemberWithName("len").GetValueAsUnsigned()
        capacity = valobj.GetChildMemberWithName("cap").GetValueAsUnsigned()
        return f"Vec(len={length}, cap={capacity})"
    except Exception as e:
        return f"Vec(<error: {e}>)"


def d_option_summary(valobj, internal_dict):
    """Summary provider for D's Option<T> type."""
    try:
        tag = valobj.GetChildMemberWithName("tag").GetValueAsUnsigned()

        if tag == 0:
            return "None"
        else:
            value = valobj.GetChildMemberWithName("value")
            value_summary = value.GetSummary() or value.GetValue() or "..."
            return f"Some({value_summary})"
    except Exception as e:
        return f"Option(<error: {e}>)"


def d_result_summary(valobj, internal_dict):
    """Summary provider for D's Result<T, E> type."""
    try:
        tag = valobj.GetChildMemberWithName("tag").GetValueAsUnsigned()

        if tag == 0:
            value = valobj.GetChildMemberWithName("ok_value")
            value_summary = value.GetSummary() or value.GetValue() or "..."
            return f"Ok({value_summary})"
        else:
            value = valobj.GetChildMemberWithName("err_value")
            value_summary = value.GetSummary() or value.GetValue() or "..."
            return f"Err({value_summary})"
    except Exception as e:
        return f"Result(<error: {e}>)"


def d_box_summary(valobj, internal_dict):
    """Summary provider for D's Box<T> type."""
    try:
        ptr = valobj.GetChildMemberWithName("ptr").GetValueAsUnsigned()

        if ptr == 0:
            return "Box(null)"

        # Dereference and get summary
        ptr_val = valobj.GetChildMemberWithName("ptr")
        inner = ptr_val.Dereference()
        inner_summary = inner.GetSummary() or inner.GetValue() or "..."
        return f"Box({inner_summary})"
    except Exception as e:
        return f"Box(<error: {e}>)"


def d_arc_summary(valobj, internal_dict):
    """Summary provider for D's Arc<T> type."""
    try:
        ptr = valobj.GetChildMemberWithName("ptr").GetValueAsUnsigned()

        if ptr == 0:
            return "Arc(null)"

        ptr_val = valobj.GetChildMemberWithName("ptr")
        inner = ptr_val.Dereference()

        try:
            strong = inner.GetChildMemberWithName("strong").GetValueAsUnsigned()
            weak = inner.GetChildMemberWithName("weak").GetValueAsUnsigned()
            value = inner.GetChildMemberWithName("value")
            value_summary = value.GetSummary() or value.GetValue() or "..."
            return f"Arc({value_summary}, strong={strong}, weak={weak})"
        except:
            return f"Arc({inner.GetSummary() or '...'})"
    except Exception as e:
        return f"Arc(<error: {e}>)"


def d_slice_summary(valobj, internal_dict):
    """Summary provider for D slices."""
    try:
        length = valobj.GetChildMemberWithName("len").GetValueAsUnsigned()
        return f"&[...; {length}]"
    except Exception as e:
        return f"&[<error: {e}>]"


def d_hashmap_summary(valobj, internal_dict):
    """Summary provider for D's HashMap<K, V> type."""
    try:
        length = valobj.GetChildMemberWithName("len").GetValueAsUnsigned()
        capacity = valobj.GetChildMemberWithName("cap").GetValueAsUnsigned()
        return f"HashMap(len={length}, cap={capacity})"
    except Exception as e:
        return f"HashMap(<error: {e}>)"


def d_future_summary(valobj, internal_dict):
    """Summary provider for D's Future state."""
    try:
        state = valobj.GetChildMemberWithName("state").GetValueAsUnsigned()
        states = {0: "Pending", 1: "Running", 2: "Ready", 3: "Cancelled"}
        return f"Future({states.get(state, f'Unknown({state})')})"
    except Exception as e:
        return f"Future(<error: {e}>)"


# =============================================================================
# Synthetic Child Providers
# =============================================================================


class DVecSyntheticProvider:
    """Synthetic child provider for Vec<T>."""

    def __init__(self, valobj, internal_dict):
        self.valobj = valobj
        self.update()

    def update(self):
        try:
            self.ptr = self.valobj.GetChildMemberWithName("ptr")
            self.len = self.valobj.GetChildMemberWithName("len").GetValueAsUnsigned()

            # Get element type
            ptr_type = self.ptr.GetType()
            if ptr_type.IsPointerType():
                self.element_type = ptr_type.GetPointeeType()
                self.element_size = self.element_type.GetByteSize()
            else:
                self.element_type = None
                self.element_size = 0
        except:
            self.len = 0
            self.element_type = None
            self.element_size = 0

    def num_children(self):
        return min(self.len, 100)

    def get_child_index(self, name):
        try:
            return int(name.lstrip("[").rstrip("]"))
        except ValueError:
            return -1

    def get_child_at_index(self, index):
        if index < 0 or index >= self.len or self.element_type is None:
            return None

        try:
            offset = index * self.element_size
            return self.ptr.CreateChildAtOffset(f"[{index}]", offset, self.element_type)
        except:
            return None

    def has_children(self):
        return self.len > 0


class DSliceSyntheticProvider:
    """Synthetic child provider for slices."""

    def __init__(self, valobj, internal_dict):
        self.valobj = valobj
        self.update()

    def update(self):
        try:
            self.ptr = self.valobj.GetChildMemberWithName("ptr")
            self.len = self.valobj.GetChildMemberWithName("len").GetValueAsUnsigned()

            ptr_type = self.ptr.GetType()
            if ptr_type.IsPointerType():
                self.element_type = ptr_type.GetPointeeType()
                self.element_size = self.element_type.GetByteSize()
            else:
                self.element_type = None
                self.element_size = 0
        except:
            self.len = 0
            self.element_type = None
            self.element_size = 0

    def num_children(self):
        return min(self.len, 50)

    def get_child_index(self, name):
        try:
            return int(name.lstrip("[").rstrip("]"))
        except ValueError:
            return -1

    def get_child_at_index(self, index):
        if index < 0 or index >= self.len or self.element_type is None:
            return None

        try:
            offset = index * self.element_size
            return self.ptr.CreateChildAtOffset(f"[{index}]", offset, self.element_type)
        except:
            return None

    def has_children(self):
        return self.len > 0


# =============================================================================
# LLDB Commands
# =============================================================================


def d_backtrace(debugger, command, result, internal_dict):
    """Enhanced backtrace with D-specific information.

    Usage: d-backtrace
    """
    target = debugger.GetSelectedTarget()
    process = target.GetProcess()
    thread = process.GetSelectedThread()

    for i, frame in enumerate(thread):
        function = frame.GetFunction()
        func_name = function.GetName() if function else frame.GetSymbol().GetName()
        if not func_name:
            func_name = "<unknown>"

        # Get location
        line_entry = frame.GetLineEntry()
        file_spec = line_entry.GetFileSpec()
        line = line_entry.GetLine()

        # Check for async markers
        is_async = "poll" in func_name.lower() or "future" in func_name.lower()
        async_marker = " [async]" if is_async else ""

        result.AppendMessage(f"#{i} {func_name}{async_marker}")
        if file_spec.IsValid():
            result.AppendMessage(f"    at {file_spec}:{line}")


def d_locals(debugger, command, result, internal_dict):
    """Show local variables with D types.

    Usage: d-locals
    """
    target = debugger.GetSelectedTarget()
    process = target.GetProcess()
    thread = process.GetSelectedThread()
    frame = thread.GetSelectedFrame()

    variables = frame.GetVariables(True, True, False, True)

    for var in variables:
        name = var.GetName()
        value = var.GetValue()
        summary = var.GetSummary()
        type_name = var.GetType().GetName()

        if summary:
            result.AppendMessage(f"{name}: {summary}")
        elif value:
            result.AppendMessage(f"{name}: {type_name} = {value}")
        else:
            result.AppendMessage(f"{name}: {type_name}")


def d_ptype(debugger, command, result, internal_dict):
    """Print detailed D type information.

    Usage: d-ptype <expression>
    """
    if not command:
        result.AppendMessage("Usage: d-ptype <expression>")
        return

    target = debugger.GetSelectedTarget()
    process = target.GetProcess()
    thread = process.GetSelectedThread()
    frame = thread.GetSelectedFrame()

    value = frame.EvaluateExpression(command)
    if not value.IsValid():
        result.AppendMessage(f"Error: Could not evaluate '{command}'")
        return

    type_obj = value.GetType()
    result.AppendMessage(f"Type: {type_obj.GetName()}")
    result.AppendMessage(f"Size: {type_obj.GetByteSize()} bytes")

    # Print fields
    num_fields = type_obj.GetNumberOfFields()
    if num_fields > 0:
        result.AppendMessage("\nFields:")
        for i in range(num_fields):
            field = type_obj.GetFieldAtIndex(i)
            offset = field.GetOffsetInBytes()
            result.AppendMessage(
                f"  +{offset:3d} {field.GetName()}: {field.GetType().GetName()}"
            )


def d_async(debugger, command, result, internal_dict):
    """Show async runtime information.

    Usage: d-async
    """
    result.AppendMessage("Async Runtime Information")
    result.AppendMessage("=" * 40)
    result.AppendMessage("(Would show active tasks, pending futures, etc.)")


# =============================================================================
# Initialization
# =============================================================================


def __lldb_init_module(debugger, internal_dict):
    """Initialize the D LLDB module."""

    # Register type summaries
    debugger.HandleCommand(
        'type summary add -F d_lldb.d_string_summary -x "^d::String$" --category demetrios'
    )
    debugger.HandleCommand(
        'type summary add -F d_lldb.d_vec_summary -x "^d::Vec<.*>$" --category demetrios'
    )
    debugger.HandleCommand(
        'type summary add -F d_lldb.d_option_summary -x "^d::Option<.*>$" --category demetrios'
    )
    debugger.HandleCommand(
        'type summary add -F d_lldb.d_result_summary -x "^d::Result<.*>$" --category demetrios'
    )
    debugger.HandleCommand(
        'type summary add -F d_lldb.d_box_summary -x "^d::Box<.*>$" --category demetrios'
    )
    debugger.HandleCommand(
        'type summary add -F d_lldb.d_arc_summary -x "^d::Arc<.*>$" --category demetrios'
    )
    debugger.HandleCommand(
        'type summary add -F d_lldb.d_slice_summary -x "^d::Slice<.*>$" --category demetrios'
    )
    debugger.HandleCommand(
        'type summary add -F d_lldb.d_hashmap_summary -x "^d::HashMap<.*>$" --category demetrios'
    )
    debugger.HandleCommand(
        'type summary add -F d_lldb.d_future_summary -x "^d::Future<.*>$" --category demetrios'
    )

    # Register synthetic providers
    debugger.HandleCommand(
        'type synthetic add -l d_lldb.DVecSyntheticProvider -x "^d::Vec<.*>$" --category demetrios'
    )
    debugger.HandleCommand(
        'type synthetic add -l d_lldb.DSliceSyntheticProvider -x "^d::Slice<.*>$" --category demetrios'
    )

    # Enable the category
    debugger.HandleCommand("type category enable demetrios")

    # Register commands
    debugger.HandleCommand("command script add -f d_lldb.d_backtrace d-backtrace")
    debugger.HandleCommand("command script add -f d_lldb.d_locals d-locals")
    debugger.HandleCommand("command script add -f d_lldb.d_ptype d-ptype")
    debugger.HandleCommand("command script add -f d_lldb.d_async d-async")
    print("Demetrios (D) LLDB support loaded.")
    print("Available commands: d-backtrace, d-locals, d-ptype, d-async")
