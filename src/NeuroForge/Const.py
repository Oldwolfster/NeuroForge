from typing import List
from typing import TYPE_CHECKING

from src.NNA.engine.TrainingRunInfo import TrainingRunInfo
# ==============================
# Global References
# ==============================
TRIs: List[TrainingRunInfo] = []
if TYPE_CHECKING:
    from src.NeuroForge.Display_Manager import DisplayManager
dm: "DisplayManager" = None  # Lazy reference to avoid circular imports

if TYPE_CHECKING:
    from src.NeuroForge.VCR             import VCR
vcr: "VCR" = None

# ==============================
# UI Constants
# ==============================
SCREEN_WIDTH  = 1900
SCREEN_HEIGHT = 900 #900

MODEL_AREA_PERCENT_LEFT     = 0.10
MODEL_AREA_PERCENT_TOP      = 0.05
MODEL_AREA_PERCENT_WIDTH    = 0.80
MODEL_AREA_PERCENT_HEIGHT   = 0.91
MODEL_AREA_PIXELS_LEFT      = SCREEN_WIDTH * MODEL_AREA_PERCENT_LEFT
MODEL_AREA_PIXELS_TOP       = SCREEN_HEIGHT * MODEL_AREA_PERCENT_TOP
MODEL_AREA_PIXELS_WIDTH     = SCREEN_WIDTH * MODEL_AREA_PERCENT_WIDTH
MODEL_AREA_PIXELS_HEIGHT    = SCREEN_HEIGHT * MODEL_AREA_PERCENT_HEIGHT


NEURON_MAX_SIZE             = 350      # Maximum width/height for a single neuron
NEURON_MIN_GAP              = 50       # Minimum space between neurons/layers
NEURON_MARGIN               = 20       # Space around visualization (currently unused)

# EZForm Field Layout
EZFORM_INPUT_BOX_HEIGHT     = 20  # Height of input boxes (was 30)
WEIGHT_MIN_HEIGHT_OUTLINE   = 3

MENU_ACTIVE                 = False
IS_RUNNING                  = True


MAX_WEIGHT      = 0.0
MAX_ACTIVATION  = 0.0
MAX_ERROR       = 0.0

# ==============================
# Pygame Objects (Initialized Later)
# ==============================
SCREEN          = None
UI_MANAGER      = None
TOOL_TIP        = None
FONT            = None
DISPLAY_MODELS  = []

# ==============================
# Popup Const
# ==============================
TOOLTIP_WIDTH_MAX   = 1669 #969 #669
TOOLTIP_HEIGHT_MAX  = 869
TOOLTIP_WIDTH       = 1169 #969 #669
TOOLTIP_HEIGHT      = 469
TOOLTIP_PLACEMENT_X =  10
TOOLTIP_PLACEMENT_Y =  10
TOOLTIP_PADDING     =   5
TOOLTIP_FONT_TITLE  =  40
TOOLTIP_FONT_SUB    =  32
TOOLTIP_FONT_HEADER =  26
TOOLTIP_FONT_BODY   =  22
TOOLTIP_COL_WIDTH   =  60  # ✅ Standardized column width
TOOLTIP_ROW_HEIGHT  =  20  # ✅ Standardized row height
TOOLTIP_HEADER_PAD  =  39  # ✅ Consistent header spacing
TOOLTIP_COND_COLUMN =   7
TOOLTIP_ADJUST_PAD  =  20

# ==============================
# Popup Divider Line Consts
# ==============================
TOOLTIP_LINE_BEFORE_BACKPROP       = 6    # After forward prop ends
TOOLTIP_LINE_AFTER_ADJUST          = 15   # After orig/new before blame calc
#TOOLTIP_LINE_BEFORE_ACTIVATION     = 6    # Before Act Gradient in fwd pass
TOOLTIP_LINE_OVER_HEADER_Y        = 0   # Y position under header row
TOOLTIP_HEADER_DIVIDER_THICKNESS   = 2
TOOLTIP_COLUMN_DIVIDER_THICKNESS   = 1

# ==============================
# UI Customization
# ==============================
JUMP_TO_EPOCH       = 0
FONT_SIZE_WEIGHT    = 22
FONT_SIZE_SMALL     = 20
#COLOR_NEURONS  = True

# ==============================
# Fonts
# ==============================
#FONT_NEURON_BANNER =  pygame.font.Font(None, 30)

# ==============================
# Colors
# ==============================
COLOR_BLACK             = (0, 0, 0)
COLOR_BLACK_NEAR        = (26, 26, 26)
COLOR_BLUE              = (50, 50, 255)
COLOR_BLUE_CORNFLOWER   = (122, 139, 214)
COLOR_BLUE_GREY         = (225, 230, 245)
COLOR_BLUE_LIGHT        = (235, 238, 248)
COLOR_BLUE_PURE         = (0, 0, 255)
COLOR_BLUE_MIDNIGHT     = (25, 25, 112)
COLOR_BLUE_SKY          = (135, 206, 235)
COLOR_BLUE_STEEL        = (70, 130, 180)
COLOR_BLUE_SUBORDINATE  = (130, 138,165)
COLOR_CREAM             = (255, 255, 200)
COLOR_CRIMSON_BU_RED    = (204, 0, 0)
COLOR_CRIMSON           = (220, 20, 60)
COLOR_CYAN              = (0, 255, 255)
COLOR_GRAY_DIM          = (105, 105, 105)
COLOR_GRAY_DARK         = (64, 64, 64)
COLOR_GREEN             = (0, 128, 0)
COLOR_GREEN_FOREST      = (34, 139, 34)
COLOR_GREEN_JADE        = (60, 179, 113)
COLOR_GREEN_KELLY       = (34, 170, 34)
COLOR_PERIWINKLE        = (170, 175,210)
COLOR_RED_FIREBRICK     = (178, 34, 34)
COLOR_RED_BURGUNDY      = (139,   0, 0)
COLOR_ORANGE            = (255, 165, 0)
COLOR_WHITE             = (255, 255, 255)
COLOR_YELLOW_BRIGHT     = (255, 215, 0)



#Below is Colors  by Purpose rather than color name.
COLOR_FOR_BANNER        = (0, 0, 255)
COLOR_FOR_SHADOW        = (30, 30, 100)  # Darker blue for depth
COLOR_FOR_POPUP         = COLOR_CREAM
COLOR_FOR_BACKGROUND    = COLOR_WHITE
COLOR_FOR_BANNER_START  = COLOR_BLUE_MIDNIGHT
COLOR_FOR_BANNER_END    = COLOR_BLUE_STEEL
COLOR_FOR_NEURON_BODY   = COLOR_BLUE_PURE
COLOR_FOR_NEURON_TEXT   = COLOR_WHITE
COLOR_FOR_BAR_GLOBAL    = COLOR_ORANGE
COLOR_FOR_BAR_SELF      = COLOR_GREEN
COLOR_FOR_ACT_POSITIVE  = COLOR_GREEN
COLOR_FOR_ACT_NEGATIVE  = COLOR_CRIMSON
COLOR_FOR_BAR1_POSITIVE = COLOR_GREEN_KELLY
COLOR_FOR_BAR1_NEGATIVE = COLOR_RED_FIREBRICK
COLOR_FOR_BAR2_POSITIVE = COLOR_GREEN_JADE
COLOR_FOR_BAR2_NEGATIVE = COLOR_RED_BURGUNDY
COLOR_FOR_HDR_BCK       = COLOR_BLUE_LIGHT
COLOR_FOR_HDR_BTM       = COLOR_PERIWINKLE
COLOR_FOR_TXT_BG        = COLOR_BLUE_GREY
COLOR_FOR_TXT_BORDER    = COLOR_BLUE_CORNFLOWER
COLOR_FOR_TXT_TXT       = COLOR_BLACK_NEAR
COLOR_FOR_TXT_PH        = COLOR_BLUE_SUBORDINATE
COLOR_eh                = (220, 255, 220)
COLOR_HIGHLIGHT_FILL    = COLOR_eh
COLOR_HIGHLIGHT_BORDER  = (218, 165, 32)
COLOR_MOLTEN            = (255, 50, 0)
COLOR_MOLTEN_GLOW       = (255, 150, 50)
COLOR_TRANSPARENT       = (0, 0, 0, 0)


COLOR_COMP_LINE_BG = (20, 30, 50)
COLOR_LABEL_BG_TRANSPARENT = (0, 0, 0, 150)
COLOR_GRAY = (128, 128, 128)
COLOR_BANNER_CORRECT = (34, 100, 34)   # Dark green - or use get_darker_color if you prefer
COLOR_BANNER_WRONG   = (139, 0, 0)     # Dark red