# BaseArena.py  NEW FILE — one return shape, gated at the boundary
from abc import ABC, abstractmethod
import numbers

class BaseArena(ABC):
    """
    Contract for every arena. Write generate_training_data(), return two things:

        return rows, field_labels                       # regression
        return rows, field_labels, outcome_labels       # classification names its classes

        rows            list of tuples. The TARGET IS LAST in every tuple.
        field_labels    one name per column, target included. Same length as a row.
        outcome_labels  optional, one name per class, in index order.

    Nothing downstream guesses. training_data() gates the shape and, when it is
    wrong, says exactly what to type instead. The FILENAME is the arena's identity
    everywhere -- picker, batch_history, scaler assignments -- so name the class
    to match the file and nothing has to go looking.
    """

    @abstractmethod
    def generate_training_data(self):
        """Build the rows. Return (rows, field_labels) or (rows, field_labels, outcome_labels)."""

    def training_data(self):
        """The one entry point. Everything downstream calls this, never the override."""
        return self.validate(self.generate_training_data())

    def validate(self, result):
        """Passes the payload through untouched, or raises something you can act on."""
        rows, fields, outcomes = self.unpack(result)
        self.check_rows(rows)
        self.check_width(rows, fields)                          # REFERENCE
        self.check_cells(rows, fields)                          # NEW
        return rows, fields, outcomes

    # BaseArena.py  method: check_cells  NEW: a nested cell dies deep in forward_pass, so catch it here
    def check_cells(self, rows, fields):
        """Every cell is a number. The forward pass multiplies cells, it cannot walk them."""
        bad = [i for i, cell in enumerate(rows[0]) if not isinstance(cell, numbers.Number)]
        if bad: raise ValueError(self.cell_message(rows[0], fields, bad))

    # BaseArena.py  method: cell_message  NEW: name the offending column, not just the type
    def cell_message(self, row, fields, bad):
        names = ", ".join(f'"{fields[i]}"' for i in bad)
        return (f"{self.name()} put a non-numeric value in column {names}.\n"
                f"    row : {str(row)[:120]}\n"
                f"TO FIX: flatten nested sequences into one column per value, "
                f"one label each.")

    # ══════════════════════════════════════════════════════════════
    # GATES
    # ══════════════════════════════════════════════════════════════

    def unpack(self, result):
        """Two forms are legal. Anything else is a missing label list."""
        if isinstance(result, tuple) and len(result) == 2:  return result[0], result[1], []
        if isinstance(result, tuple) and len(result) == 3:  return result
        raise ValueError(self.shape_message(result))

    def check_rows(self, rows):
        """No rows means the loop never ran or the filter ate everything."""
        if isinstance(rows, list) and rows: return
        raise ValueError(self.rows_message(rows))

    def check_width(self, rows, fields):
        """One label per column, target included."""
        width = len(rows[0])
        if len(fields) == width: return
        raise ValueError(self.width_message(rows[0], fields, width))

    # ══════════════════════════════════════════════════════════════
    # MESSAGES — every failure names the fix
    # ══════════════════════════════════════════════════════════════

    def shape_message(self, result):
        return (f"{self.name()}.generate_training_data() returned {type(result).__name__}, "
                f"not (rows, labels).\n"
                f"TO FIX: end the method with a label list, target last --\n"
                f'    return training_data, ["Feature A", "Feature B", "Target"]')

    def rows_message(self, rows):
        return (f"{self.name()}.generate_training_data() produced no rows ({type(rows).__name__}).\n"
                f"TO FIX: check the sample loop and any row filter -- dropping every row on a "
                f"missing value is the usual cause.")

    def width_message(self, row, fields, width):
        return (f"{self.name()} labelled {len(fields)} columns but each row has {width}.\n"
                f"    row    : {row}\n"
                f"    labels : {fields}\n"
                f"TO FIX: one label per column, target last.")

    def name(self):
        return type(self).__name__