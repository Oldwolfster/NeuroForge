from src.NeuroForge import Const
from src.NeuroForge.DisplayModel import DisplayModel


class ModelGenerator:
    display_models = []

    @staticmethod
    def create_models():
        ModelGenerator.display_models = []
        positions = ModelGenerator.calculate_positions()

        for TRI in Const.TRIs:
            position = positions[TRI.run_id]
            model = DisplayModel(TRI, position)
            model.initialize_with_model_info()
            ModelGenerator.display_models.append(model)

        return ModelGenerator.display_models

    @staticmethod
    def calculate_positions():
        """1 model = full area. 2 models = top/bottom split."""
        positions = {}

        if len(Const.TRIs) == 1:
            positions[Const.TRIs[0].run_id] = {
                "left": Const.MODEL_AREA_PIXELS_LEFT,
                "top": Const.MODEL_AREA_PIXELS_TOP,
                "width": Const.MODEL_AREA_PIXELS_WIDTH,
                "height": Const.MODEL_AREA_PIXELS_HEIGHT,
            }
        else:
            half_height = Const.MODEL_AREA_PIXELS_HEIGHT // 2
            positions[Const.TRIs[0].run_id] = {
                "left": Const.MODEL_AREA_PIXELS_LEFT,
                "top": Const.MODEL_AREA_PIXELS_TOP,
                "width": Const.MODEL_AREA_PIXELS_WIDTH,
                "height": half_height,
            }
            positions[Const.TRIs[1].run_id] = {
                "left": Const.MODEL_AREA_PIXELS_LEFT,
                "top": Const.MODEL_AREA_PIXELS_TOP + half_height,
                "width": Const.MODEL_AREA_PIXELS_WIDTH,
                "height": half_height,
            }

        return positions