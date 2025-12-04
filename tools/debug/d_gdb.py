#!/usr/bin/env python3
"""GDB pretty printers for Demetrios (D) types.

This module provides GDB pretty printers and custom commands for debugging
D programs. Load it in GDB with:

    source /path/to/d_gdb.py

Or add to your ~/.gdbinit:

    python
    import sys
    sys.path.insert(0, '/path/to/demetrios/tools/debug')
    import d_gdb
    end
"""

import gdb
import re


# =============================================================================
# Pretty Printers
# =============================================================================

class DStringPrinter:
    """Pretty printer for D's String type."""

    def __init__(self, val):
        self.val = val

    def to_string(self):
        try:
            ptr = self.val['ptr']
            length = int(self.val['len'])
            capacity = int(self.val['cap'])

            if ptr == 0:
                return "String(null)"

            # Read string data
            inferior = gdb.selected_inferior()
            data = inferior.read_memory(ptr, length).tobytes().decode('utf-8', errors='replace')

            if len(data) > 100:
                data = data[:100] + "..."

            return f'String("{data}", len={length}, cap={capacity})'
        except Exception as e:
            return f"String(<error: {e}>)"


class DVecPrinter:
    """Pretty printer for D's Vec<T> type."""

    def __init__(self, val):
        self.val = val

    def to_string(self):
        try:
            length = int(self.val['len'])
            capacity = int(self.val['cap'])
            return f"Vec(len={length}, cap={capacity})"
        except Exception as e:
            return f"Vec(<error: {e}>)"

    def children(self):
        try:
            ptr = self.val['ptr']
            length = int(self.val['len'])

            for i in range(min(length, 100)):  # Limit to 100 elements
                yield f'[{i}]', (ptr + i).dereference()

            if length > 100:
                yield '...', f'({length - 100} more elements)'
        except Exception:
            pass

    def display_hint(self):
        return 'array'


class DOptionPrinter:
    """Pretty printer for D's Option<T> type."""

    def __init__(self, val):
        self.val = val

    def to_string(self):
        try:
            # Try tagged union format
            tag = int(self.val['tag'])

            if tag == 0:  # None
                return "None"
            else:  # Some
                try:
                    value = self.val['value']
                    return f"Some({value})"
                except:
                    return "Some(...)"
        except Exception as e:
            return f"Option(<error: {e}>)"


class DResultPrinter:
    """Pretty printer for D's Result<T, E> type."""

    def __init__(self, val):
        self.val = val

    def to_string(self):
        try:
            tag = int(self.val['tag'])

            if tag == 0:  # Ok
                try:
                    value = self.val['ok_value']
                    return f"Ok({value})"
                except:
                    return "Ok(...)"
            else:  # Err
                try:
                    value = self.val['err_value']
                    return f"Err({value})"
                except:
                    return "Err(...)"
        except Exception as e:
            return f"Result(<error: {e}>)"


class DHashMapPrinter:
    """Pretty printer for D's HashMap<K, V> type."""

    def __init__(self, val):
        self.val = val

    def to_string(self):
        try:
            length = int(self.val['len'])
            capacity = int(self.val['cap'])
            return f"HashMap(len={length}, cap={capacity})"
        except Exception as e:
            return f"HashMap(<error: {e}>)"

    def children(self):
        # Implementation would iterate through buckets
        pass

    def display_hint(self):
        return 'map'


class DBoxPrinter:
    """Pretty printer for D's Box<T> type."""

    def __init__(self, val):
        self.val = val

    def to_string(self):
        try:
            ptr = self.val['ptr']

            if ptr == 0:
                return "Box(null)"

            inner = ptr.dereference()
            return f"Box({inner})"
        except Exception as e:
            return f"Box(<error: {e}>)"


class DArcPrinter:
    """Pretty printer for D's Arc<T> type."""

    def __init__(self, val):
        self.val = val

    def to_string(self):
        try:
            ptr = self.val['ptr']

            if ptr == 0:
                return "Arc(null)"

            # Get reference count
            inner = ptr.dereference()
            try:
                strong = int(inner['strong'])
                weak = int(inner['weak'])
                value = inner['value']
                return f"Arc({value}, strong={strong}, weak={weak})"
            except:
                return f"Arc({inner})"
        except Exception as e:
            return f"Arc(<error: {e}>)"


class DFuturePrinter:
    """Pretty printer for D's Future state."""

    def __init__(self, val):
        self.val = val

    def to_string(self):
        try:
            state = int(self.val['state'])
            states = {0: 'Pending', 1: 'Running', 2: 'Ready', 3: 'Cancelled'}
            return f"Future({states.get(state, f'Unknown({state})')})"
        except Exception as e:
            return f"Future(<error: {e}>)"


class DSlicePrinter:
    """Pretty printer for D slices."""

    def __init__(self, val):
        self.val = val

    def to_string(self):
        try:
            ptr = self.val['ptr']
            length = int(self.val['len'])
            return f"&[...; {length}]"
        except Exception as e:
            return f"&[<error: {e}>]"

    def children(self):
        try:
            ptr = self.val['ptr']
            length = int(self.val['len'])

            for i in range(min(length, 50)):
                yield f'[{i}]', (ptr + i).dereference()

            if length > 50:
                yield '...', f'({length - 50} more elements)'
        except Exception:
            pass

    def display_hint(self):
        return 'array'


class DTaskPrinter:
    """Pretty printer for D async Task."""

    def __init__(self, val):
        self.val = val

    def to_string(self):
        try:
            task_id = int(self.val['id'])
            state = int(self.val['state'])
            states = {0: 'Ready', 1: 'Running', 2: 'Blocked', 3: 'Completed', 4: 'Cancelled'}
            return f"Task(id={task_id}, state={states.get(state, 'Unknown')})"
        except Exception as e:
            return f"Task(<error: {e}>)"


class DSpanPrinter:
    """Pretty printer for D tracing Span."""

    def __init__(self, val):
        self.val = val

    def to_string(self):
        try:
            name = self.val['name']
            span_id = int(self.val['id'])
            return f"Span(\"{name}\", id={span_id:x})"
        except Exception as e:
            return f"Span(<error: {e}>)"


# =============================================================================
# Type Recognizer
# =============================================================================

def d_type_lookup(val):
    """Look up the appropriate pretty printer for a D type."""
    try:
        type_name = str(val.type.strip_typedefs())
    except:
        return None

    # Handle qualified names
    if '::' in type_name:
        type_name = type_name.split('::')[-1]

    # Remove template parameters for matching
    base_type = re.sub(r'<.*>', '', type_name)

    printers = {
        'String': DStringPrinter,
        'Vec': DVecPrinter,
        'Option': DOptionPrinter,
        'Result': DResultPrinter,
        'HashMap': DHashMapPrinter,
        'HashSet': DHashMapPrinter,
        'Box': DBoxPrinter,
        'Arc': DArcPrinter,
        'Rc': DArcPrinter,
        'Future': DFuturePrinter,
        'Slice': DSlicePrinter,
        'Task': DTaskPrinter,
        'Span': DSpanPrinter,
    }

    if base_type in printers:
        return printers[base_type](val)

    return None


# Register pretty printers
gdb.pretty_printers.append(d_type_lookup)


# =============================================================================
# Custom GDB Commands
# =============================================================================

class DBacktrace(gdb.Command):
    """Enhanced backtrace with D-specific information.

    Usage: d-backtrace [full]

    Shows backtrace with D-specific annotations including:
    - Effect annotations for each function
    - Linear type state
    - Async task context
    """

    def __init__(self):
        super().__init__("d-backtrace", gdb.COMMAND_STACK)

    def invoke(self, arg, from_tty):
        show_full = 'full' in arg.lower()
        frame = gdb.newest_frame()
        frame_num = 0

        while frame:
            sal = frame.find_sal()
            name = frame.name() or "<unknown>"

            # Try to get D-specific info
            effects = self._get_effects(name)
            is_async = self._is_async_frame(name)

            # Print frame info
            async_marker = " [async]" if is_async else ""
            print(f"#{frame_num} {name}{async_marker}")

            if sal.symtab:
                print(f"    at {sal.symtab.filename}:{sal.line}")

            if effects:
                print(f"    effects: {', '.join(effects)}")

            if show_full:
                self._print_locals(frame)

            frame = frame.older()
            frame_num += 1

    def _get_effects(self, func_name):
        """Look up effects from debug info."""
        # Would look up effects from DWARF custom attributes
        # For now, detect common patterns
        effects = []
        if 'io' in func_name.lower() or 'print' in func_name.lower():
            effects.append('IO')
        if 'alloc' in func_name.lower() or 'vec' in func_name.lower():
            effects.append('Alloc')
        if 'async' in func_name.lower() or 'spawn' in func_name.lower():
            effects.append('Async')
        return effects

    def _is_async_frame(self, func_name):
        """Check if this is an async frame."""
        return 'poll' in func_name.lower() or 'future' in func_name.lower()

    def _print_locals(self, frame):
        """Print local variables in frame."""
        try:
            block = frame.block()
            while block:
                for sym in block:
                    if sym.is_variable:
                        try:
                            val = frame.read_var(sym)
                            printer = d_type_lookup(val)
                            if printer:
                                print(f"      {sym.name}: {printer.to_string()}")
                            else:
                                print(f"      {sym.name}: {val}")
                        except:
                            pass
                block = block.superblock
        except:
            pass


class DLocals(gdb.Command):
    """Show local variables with D types.

    Usage: d-locals

    Displays local variables in the current frame using D-aware pretty printers.
    """

    def __init__(self):
        super().__init__("d-locals", gdb.COMMAND_DATA)

    def invoke(self, arg, from_tty):
        frame = gdb.selected_frame()

        try:
            block = frame.block()
        except RuntimeError:
            print("No symbol table for this frame.")
            return

        printed = set()

        while block:
            for sym in block:
                if sym.is_variable or sym.is_argument:
                    if sym.name in printed:
                        continue
                    printed.add(sym.name)

                    try:
                        val = frame.read_var(sym)
                        printer = d_type_lookup(val)

                        kind = "arg" if sym.is_argument else "var"
                        if printer:
                            print(f"[{kind}] {sym.name}: {printer.to_string()}")
                        else:
                            print(f"[{kind}] {sym.name}: {val}")
                    except Exception as e:
                        print(f"[{kind}] {sym.name}: <error: {e}>")

            block = block.superblock


class DPrintType(gdb.Command):
    """Print detailed D type information.

    Usage: d-ptype <expression>

    Shows detailed type information including:
    - Type name and size
    - Fields for structs/enums
    - Effect annotations
    - Linear/affine qualifiers
    """

    def __init__(self):
        super().__init__("d-ptype", gdb.COMMAND_DATA)

    def invoke(self, arg, from_tty):
        if not arg:
            print("Usage: d-ptype <expression>")
            return

        try:
            val = gdb.parse_and_eval(arg)
            type_obj = val.type.strip_typedefs()
            type_name = str(type_obj)

            print(f"Type: {type_name}")
            print(f"Size: {type_obj.sizeof} bytes")

            # Check for linear/affine markers
            if 'linear' in type_name.lower():
                print("Qualifier: linear (must be used exactly once)")
            elif 'affine' in type_name.lower():
                print("Qualifier: affine (must be used at most once)")

            # Print fields for structs
            if type_obj.code == gdb.TYPE_CODE_STRUCT:
                print("\nFields:")
                for field in type_obj.fields():
                    offset = field.bitpos // 8 if hasattr(field, 'bitpos') else 0
                    print(f"  +{offset:3d} {field.name}: {field.type}")

            # Print variants for enums
            elif type_obj.code == gdb.TYPE_CODE_ENUM:
                print("\nVariants:")
                for field in type_obj.fields():
                    print(f"  {field.name} = {field.enumval}")

        except Exception as e:
            print(f"Error: {e}")


class DAsyncInfo(gdb.Command):
    """Show async runtime information.

    Usage: d-async [tasks|executors|channels]

    Displays async runtime state including:
    - Active tasks and their states
    - Executor information
    - Channel statistics
    """

    def __init__(self):
        super().__init__("d-async", gdb.COMMAND_DATA)

    def invoke(self, arg, from_tty):
        print("Async Runtime Information")
        print("=" * 40)

        # Try to find async runtime globals
        try:
            runtime = gdb.parse_and_eval("&ASYNC_RUNTIME")
            print(f"Runtime: {runtime}")

            # Try to get task count
            try:
                task_count = gdb.parse_and_eval("ASYNC_RUNTIME.task_count")
                print(f"Active tasks: {task_count}")
            except:
                pass

        except:
            print("(Could not access async runtime state)")
            print("Make sure the program is linked with debug symbols.")


class DEffects(gdb.Command):
    """Show effect information for a function.

    Usage: d-effects [function_name]

    If no function name is given, shows effects for the current frame.
    """

    def __init__(self):
        super().__init__("d-effects", gdb.COMMAND_DATA)

    def invoke(self, arg, from_tty):
        if arg:
            func_name = arg
        else:
            frame = gdb.selected_frame()
            func_name = frame.name() or "<unknown>"

        print(f"Effects for {func_name}:")

        # Would look up from DWARF custom attributes
        # For demo, show detected effects
        effects = []
        if 'io' in func_name.lower() or 'print' in func_name.lower():
            effects.append('IO')
        if 'alloc' in func_name.lower() or 'new' in func_name.lower():
            effects.append('Alloc')
        if 'panic' in func_name.lower():
            effects.append('Panic')
        if 'async' in func_name.lower():
            effects.append('Async')

        if effects:
            print(f"  with {', '.join(effects)}")
        else:
            print("  (no effects detected)")


class DOwnership(gdb.Command):
    """Show ownership information for a value.

    Usage: d-ownership <expression>

    Shows ownership state including:
    - Whether the value is moved
    - Reference counts (for Arc/Rc)
    - Borrow state
    """

    def __init__(self):
        super().__init__("d-ownership", gdb.COMMAND_DATA)

    def invoke(self, arg, from_tty):
        if not arg:
            print("Usage: d-ownership <expression>")
            return

        try:
            val = gdb.parse_and_eval(arg)
            type_name = str(val.type)

            print(f"Ownership info for: {arg}")
            print(f"Type: {type_name}")

            # Check for reference counting types
            if 'Arc' in type_name or 'Rc' in type_name:
                try:
                    ptr = val['ptr']
                    if ptr != 0:
                        inner = ptr.dereference()
                        strong = int(inner['strong'])
                        weak = int(inner['weak'])
                        print(f"Reference counts: strong={strong}, weak={weak}")
                except:
                    pass

            # Check for box types
            elif 'Box' in type_name:
                try:
                    ptr = val['ptr']
                    if ptr == 0:
                        print("Status: null (dropped or moved)")
                    else:
                        print("Status: owned")
                except:
                    pass

            # Check for linear/affine markers
            if 'linear' in type_name.lower():
                print("Ownership: linear (must be consumed)")
            elif 'affine' in type_name.lower():
                print("Ownership: affine (may be dropped)")
            else:
                print("Ownership: normal (Copy or implicit Drop)")

        except Exception as e:
            print(f"Error: {e}")


# =============================================================================
# Breakpoint Helpers
# =============================================================================

class DPanicBreakpoint(gdb.Breakpoint):
    """Breakpoint that triggers on D panic."""

    def __init__(self):
        # Try various panic function names
        for name in ['d_panic', '_D_panic', 'std::panic::panic', 'panic']:
            try:
                super().__init__(name, gdb.BP_BREAKPOINT)
                print(f"Set panic breakpoint on {name}")
                return
            except:
                continue
        print("Warning: Could not set panic breakpoint")

    def stop(self):
        print("\n*** D PANIC ***")
        # Try to print panic message
        try:
            msg = gdb.parse_and_eval("$rdi")  # First argument on x86_64
            print(f"Message: {msg}")
        except:
            pass
        return True  # Stop execution


class DSetPanicBreak(gdb.Command):
    """Set a breakpoint on D panic.

    Usage: d-panic-break
    """

    def __init__(self):
        super().__init__("d-panic-break", gdb.COMMAND_BREAKPOINTS)

    def invoke(self, arg, from_tty):
        DPanicBreakpoint()


# =============================================================================
# Register Commands
# =============================================================================

DBacktrace()
DLocals()
DPrintType()
DAsyncInfo()
DEffects()
DOwnership()
DSetPanicBreak()

print("Demetrios (D) GDB support loaded.")
print("Available commands: d-backtrace, d-locals, d-ptype, d-async, d-effects, d-ownership, d-panic-break")
