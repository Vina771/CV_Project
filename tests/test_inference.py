from app.utils.inference import count_detections, run_image_detection


class FakeModel:
    def predict(self, **kwargs):
        self.kwargs = kwargs
        return [FakeResult()]


class FakeBoxes:
    def __len__(self):
        return 2


class FakeResult:
    boxes = FakeBoxes()


def test_run_image_detection_returns_first_result():
    model = FakeModel()

    result = run_image_detection(model, "image.jpg", confidence=0.4, image_size=640)

    assert isinstance(result, FakeResult)
    assert model.kwargs["source"] == "image.jpg"
    assert model.kwargs["conf"] == 0.4
    assert model.kwargs["imgsz"] == 640


def test_count_detections_uses_boxes_length():
    assert count_detections(FakeResult()) == 2
