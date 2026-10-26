# ══════════════════════════════════════════════════════════════════════════════
# db_catalog.py     NEW FILE — populates the lego table from source code
#                   lifted from db_create.py, gladiator/arena paths collapsed
# ══════════════════════════════════════════════════════════════════════════════
import re
import sys
from pathlib import Path


class db_catalog:
    """Scrapes legos, scalers, gladiators and arenas out of source and into `lego`.

       populate_all() is the only entry point.  It no-ops if the table already
       has rows, so it is safe on every process start.
    """

    # db_catalog.py  method: populate_all  NEW:  sole entry point, no-ops if populated
    @staticmethod
    def populate_all(db):
        if db.query_value("SELECT COUNT(*) FROM lego"): return
        db_catalog.populate_legos(db)
        db_catalog.populate_scalers(db)
        db_catalog.populate_folder(db, "gladiators", "gladiator")
        db_catalog.populate_folder(db, "arenas",     "arena")

    # db_catalog.py  method: populate_legos  NEW:  every LegoBase instance
    @staticmethod
    def populate_legos(db):
        from src.NNA.legos._LegoManager import LegoManager
        for lego in LegoManager().lego_instances:
            db_catalog.insert_lego_row(db,
                name        = lego.name,
                dimensions  = ", ".join(lego.applicable_dimensions),
                desc        = getattr(lego, 'desc',        ''),
                when_to_use = getattr(lego, 'when_to_use', ''),
                best_for    = getattr(lego, 'best_for',    ''),
                code        = db_catalog.extract_lego_code(lego))

    # db_catalog.py  method: populate_scalers  NEW:  scan Scaler.py for Scaler_XXX
    @staticmethod
    def populate_scalers(db):
        scaler_path = Path(__file__).parent.parent / "legos" / "Scaler.py"
        lines       = scaler_path.read_text(encoding='utf-8').splitlines()

        for i, line in enumerate(lines):
            if not re.match(r'^Scaler_\w+\s*=\s*Scaler\(', line): continue
            inst_block  = db_catalog.extract_block(lines, i)
            class_block = db_catalog.find_class_above(lines, i)
            code        = f"{class_block}\n\n{inst_block}" if class_block else inst_block

            db_catalog.insert_lego_row(db,
                name        = db_catalog.extract_kwarg_string(inst_block, 'name'),
                dimensions  = "input_scalers, target_scaler",
                desc        = db_catalog.extract_kwarg_string(inst_block, 'desc'),
                when_to_use = db_catalog.extract_kwarg_string(inst_block, 'when_to_use'),
                best_for    = db_catalog.extract_kwarg_string(inst_block, 'best_for'),
                code        = code)

    # db_catalog.py  method: populate_folder  NEW:  was populate_gladiators + populate_arenas
    @staticmethod
    def populate_folder(db, subfolder, dimension):
        folder = db_catalog.find_coliseum_dir(subfolder)
        for filepath in sorted(folder.glob("*.py")):
            if filepath.name.startswith("__"): continue
            db_catalog.insert_lego_row(db,
                name        = filepath.stem,
                dimensions  = dimension,
                desc        = db_catalog.extract_class_docstring(filepath),
                when_to_use = "",
                best_for    = "",
                code        = filepath.read_text(encoding='utf-8'))

    # db_catalog.py  method: insert_lego_row  NEW:  one row
    @staticmethod
    def insert_lego_row(db, name, dimensions, desc, when_to_use, best_for, code):
        db.execute("""
            INSERT INTO lego (name, dimensions, desc, when_to_use, best_for, code)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (name, dimensions, desc, when_to_use, best_for, code))

    # db_catalog.py  method: find_coliseum_dir  NEW:  walk up for coliseum/<subfolder>
    @staticmethod
    def find_coliseum_dir(subfolder):
        current = Path(__file__).resolve().parent
        for _ in range(5):
            candidate = current / "coliseum" / subfolder
            if candidate.is_dir(): return candidate
            current = current.parent
        raise FileNotFoundError(f"Could not find coliseum/{subfolder}/ directory")

    # ══════════════════════════════════════════════════════════════════════════
    #   SOURCE EXTRACTION
    # ══════════════════════════════════════════════════════════════════════════

    # db_catalog.py  method: extract_lego_code  NEW:  instantiation + referenced funcs
    @staticmethod
    def extract_lego_code(lego):
        source_file = db_catalog.get_lego_source_file(lego)
        if not source_file: return "(source not found)"

        lines     = Path(source_file).read_text(encoding='utf-8').splitlines()
        inst_line = db_catalog.find_instantiation_line(lines, lego.STAMPED_name)
        if inst_line is None: return "(instantiation not found)"

        inst_block = db_catalog.extract_block(lines, inst_line)
        fn_names   = db_catalog.extract_referenced_names(inst_block, lines, inst_line)
        fn_blocks  = [db_catalog.find_function_above(lines, inst_line, fn) for fn in fn_names]
        return "\n\n".join([b for b in fn_blocks if b] + [inst_block])

    # db_catalog.py  method: get_lego_source_file  NEW:  module path for a lego
    @staticmethod
    def get_lego_source_file(lego):
        module = sys.modules.get(type(lego).__module__)
        return getattr(module, '__file__', None) if module else None

    # db_catalog.py  method: find_instantiation_line  NEW:  'Name = Strategy...'
    @staticmethod
    def find_instantiation_line(lines, stamped_name):
        pattern = re.compile(rf'^{re.escape(stamped_name)}\s*=\s*')
        for i, line in enumerate(lines):
            if pattern.match(line): return i
        return None

    # db_catalog.py  method: extract_block  NEW:  multi-line statement until parens balance
    @staticmethod
    def extract_block(lines, start_line):
        block, depth, started = [], 0, False
        for i in range(start_line, len(lines)):
            line = lines[i]
            block.append(line)
            depth += line.count('(') - line.count(')')
            if line.count('(') > 0: started = True
            if started and depth <= 0: break
        return "\n".join(block)

    # db_catalog.py  method: extract_referenced_names  NEW:  kwarg values that are defined names
    @staticmethod
    def extract_referenced_names(inst_block, lines, inst_line):
        skip_values = ('True', 'False', 'None', 'self')
        skip_kwargs = ('name', 'short_name', 'desc', 'when_to_use', 'best_for',
                       'popup_formula', 'derivative_formula')
        names = []
        for match in re.finditer(r'(\w+)\s*=\s*(\w+)', inst_block):
            kwarg_name, kwarg_value = match.group(1), match.group(2)
            if kwarg_value in skip_values:                              continue
            if re.match(r'^\d', kwarg_value):                           continue
            if kwarg_name in skip_kwargs:                               continue
            if db_catalog.is_defined_above(lines, inst_line, kwarg_value):
                names.append(kwarg_value)
        return names

    # db_catalog.py  method: is_defined_above  NEW:  'def name' or 'class name' above
    @staticmethod
    def is_defined_above(lines, before_line, name):
        pattern = re.compile(rf'^(def|class)\s+{re.escape(name)}\b')
        return any(pattern.match(lines[i]) for i in range(before_line - 1, -1, -1))

    # db_catalog.py  method: find_function_above  NEW:  locate then extract a def
    @staticmethod
    def find_function_above(lines, before_line, func_name):
        pattern = re.compile(rf'^def\s+{re.escape(func_name)}\s*\(')
        for i in range(before_line - 1, -1, -1):
            if pattern.match(lines[i]): return db_catalog.extract_function_block(lines, i)
        return None

    # db_catalog.py  method: extract_function_block  NEW:  def through last indented line
    @staticmethod
    def extract_function_block(lines, start_line):
        block = [lines[start_line]]
        for i in range(start_line + 1, len(lines)):
            line = lines[i]
            if line.strip() == '':          block.append(line);     continue
            if line[0] in (' ', '\t'):      block.append(line)
            else:                           break
        return "\n".join(db_catalog.trim_trailing_blanks(block))

    # db_catalog.py  method: find_class_above  NEW:  class immediately above a line
    @staticmethod
    def find_class_above(lines, before_line):
        for i in range(before_line - 1, -1, -1):
            if lines[i].strip() == '': continue
            if re.match(r'^class\s+\w+', lines[i]):
                return db_catalog.extract_class_block(lines, i, before_line)
            break
        return None

    # db_catalog.py  method: extract_class_block  NEW:  start_line up to stop_before
    @staticmethod
    def extract_class_block(lines, start_line, stop_before):
        block = list(lines[start_line:stop_before])
        return "\n".join(db_catalog.trim_trailing_blanks(block))

    # db_catalog.py  method: trim_trailing_blanks  NEW:  shared by both extractors
    @staticmethod
    def trim_trailing_blanks(block):
        while block and block[-1].strip() == '': block.pop()
        return block

    # db_catalog.py  method: extract_kwarg_string  NEW:  name='Foo' -> Foo
    @staticmethod
    def extract_kwarg_string(inst_block, kwarg_name):
        match = re.search(rf'{kwarg_name}\s*=\s*["\'](.+?)["\']', inst_block)
        return match.group(1) if match else ''

    # db_catalog.py  method: extract_class_docstring  NEW:  first class docstring in a file
    @staticmethod
    def extract_class_docstring(filepath):
        lines    = filepath.read_text(encoding='utf-8').splitlines()
        in_class = False
        for i, line in enumerate(lines):
            if re.match(r'^class\s+\w+', line):     in_class = True;    continue
            if in_class and '"""' in line:          return db_catalog.read_docstring(lines, i)
            if in_class and line.strip() and not line.strip().startswith('#'): break
        return ""

    # db_catalog.py  method: read_docstring  NEW:  single or multi-line, from opening quotes
    @staticmethod
    def read_docstring(lines, start_line):
        single = re.search(r'"""(.+?)"""', lines[start_line])
        if single: return single.group(1).strip()

        doc = [lines[start_line].split('"""', 1)[1]]
        for j in range(start_line + 1, len(lines)):
            if '"""' in lines[j]:
                doc.append(lines[j].split('"""')[0])
                break
            doc.append(lines[j])
        return "\n".join(doc).strip()

