import unittest

from speech_to_text_app.hardware import CudaStatus, probe_cuda_status


class HardwareStatusTests(unittest.TestCase):
    def test_reports_cuda_ready_when_float16_is_supported(self):
        status = probe_cuda_status(
            device_count_getter=lambda: 1,
            compute_types_getter=lambda device: {"float16", "int8_float16"},
        )

        self.assertEqual(
            CudaStatus(
                available=True,
                device_count=1,
                supported_compute_types=["float16", "int8_float16"],
                error=None,
            ),
            status,
        )

    def test_reports_unavailable_when_no_cuda_device_exists(self):
        status = probe_cuda_status(
            device_count_getter=lambda: 0,
            compute_types_getter=lambda device: {"float16"},
        )

        self.assertFalse(status.available)
        self.assertEqual(0, status.device_count)
        self.assertEqual("No CUDA devices detected", status.error)

    def test_reports_probe_errors_without_raising(self):
        def failing_device_count():
            raise RuntimeError("driver unavailable")

        status = probe_cuda_status(
            device_count_getter=failing_device_count,
            compute_types_getter=lambda device: {"float16"},
        )

        self.assertFalse(status.available)
        self.assertEqual(0, status.device_count)
        self.assertEqual([], status.supported_compute_types)
        self.assertEqual("driver unavailable", status.error)


if __name__ == "__main__":
    unittest.main()
