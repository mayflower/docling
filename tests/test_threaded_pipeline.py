import logging
import time
from pathlib import Path
from threading import Event, Thread
from typing import List

import pytest

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import ConversionStatus, InputFormat, Page
from docling.datamodel.document import ConversionResult, InputDocument
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    ThreadedPdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.standard_pdf_pipeline import (
    StandardPdfPipeline,
    ThreadedItem,
    ThreadedPipelineStage,
    ThreadedQueue,
)
from docling.pipeline.threaded_standard_pdf_pipeline import ThreadedStandardPdfPipeline

pytestmark = pytest.mark.ml_pdf_model


def test_stage_shutdown_keeps_pdf_alive_until_inflight_model_returns():
    """Timeout cleanup must not unload PDFium while a model still uses its page."""
    entered = Event()
    release = Event()
    stopped = Event()
    sizes = []
    in_doc = InputDocument(
        path_or_stream=Path("tests/data/pdf/2305.03393v1-pg9.pdf"),
        format=InputFormat.PDF,
        backend=PyPdfiumDocumentBackend,
    )
    page = Page(page_no=1)
    page._backend = in_doc._backend.load_page(0)

    def model(conv_res, pages):
        entered.set()
        assert release.wait(30), "test did not release the in-flight model"
        sizes.append(page._backend.get_size())
        return pages

    stage = ThreadedPipelineStage(
        name="slow-ocr",
        model=model,
        batch_size=1,
        batch_timeout=0.01,
        queue_max_size=1,
    )
    stage.start()
    stage.input_queue.put(
        ThreadedItem(
            payload=page,
            run_id=1,
            page_no=1,
            conv_res=ConversionResult(input=in_doc),
        )
    )
    assert entered.wait(5)

    def stop_and_unload():
        stage.stop()
        page._backend.unload()
        in_doc._backend.unload()
        stopped.set()

    cleanup = Thread(target=stop_and_unload)
    cleanup.start()
    try:
        # Exceed the previous 15-second join grace period. A real running
        # model, not a mocked thread, must retain its PDF resource throughout.
        assert not stopped.wait(16)
    finally:
        release.set()
        cleanup.join(5)
    assert stopped.is_set()
    assert len(sizes) == 1
    assert sizes[0].width > 0


def test_stage_shutdown_unblocks_a_full_output_queue():
    processed = Event()
    stopped = Event()
    in_doc = InputDocument(
        path_or_stream=Path("tests/data/pdf/2305.03393v1-pg9.pdf"),
        format=InputFormat.PDF,
        backend=PyPdfiumDocumentBackend,
    )
    item = ThreadedItem(
        payload=Page(page_no=1),
        run_id=1,
        page_no=1,
        conv_res=ConversionResult(input=in_doc),
    )

    def model(conv_res, pages):
        processed.set()
        return pages

    output = ThreadedQueue(1)
    assert output.put(item)
    stage = ThreadedPipelineStage(
        name="full-output",
        model=model,
        batch_size=1,
        batch_timeout=0.01,
        queue_max_size=1,
    )
    stage.add_output_queue(output)
    stage.start()
    stage.input_queue.put(item)
    assert processed.wait(5)

    def stop():
        stage.stop()
        stopped.set()

    cleanup = Thread(target=stop)
    cleanup.start()
    try:
        assert stopped.wait(2), "shutdown is blocked by the undrained output queue"
    finally:
        output.close()
        cleanup.join(5)
        in_doc._backend.unload()


def test_threaded_pipeline_multiple_documents():
    """Test threaded pipeline with multiple documents and compare with standard pipeline"""
    test_files = [
        "tests/data/pdf/2203.01017v2.pdf",
        "tests/data/pdf/2206.01062.pdf",
        "tests/data/pdf/2305.03393v1.pdf",
    ]
    # test_files = [str(f) for f in Path("test/data/pdf").rglob("*.pdf")]

    do_ts = False
    do_ocr = False

    run_threaded = True
    run_serial = True

    if run_threaded:
        # Threaded pipeline
        threaded_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=ThreadedStandardPdfPipeline,
                    pipeline_options=ThreadedPdfPipelineOptions(
                        layout_batch_size=1,
                        table_batch_size=1,
                        ocr_batch_size=1,
                        batch_polling_interval_seconds=1.0,
                        do_table_structure=do_ts,
                        do_ocr=do_ocr,
                    ),
                )
            }
        )

        threaded_converter.initialize_pipeline(InputFormat.PDF)

        # Test threaded pipeline
        threaded_success_count = 0
        threaded_failure_count = 0
        start_time = time.perf_counter()
        for result in threaded_converter.convert_all(test_files, raises_on_error=True):
            print(
                "Finished converting document with threaded pipeline:",
                result.input.file.name,
            )
            if result.status == ConversionStatus.SUCCESS:
                threaded_success_count += 1
            else:
                threaded_failure_count += 1
        threaded_time = time.perf_counter() - start_time

        del threaded_converter

        print(f"Threaded pipeline:  {threaded_time:.2f} seconds")

    if run_serial:
        # Standard pipeline
        standard_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=StandardPdfPipeline,
                    pipeline_options=PdfPipelineOptions(
                        do_table_structure=do_ts,
                        do_ocr=do_ocr,
                    ),
                )
            }
        )

        standard_converter.initialize_pipeline(InputFormat.PDF)

        # Test standard pipeline
        standard_success_count = 0
        standard_failure_count = 0
        start_time = time.perf_counter()
        for result in standard_converter.convert_all(test_files, raises_on_error=True):
            print(
                "Finished converting document with standard pipeline:",
                result.input.file.name,
            )
            if result.status == ConversionStatus.SUCCESS:
                standard_success_count += 1
            else:
                standard_failure_count += 1
        standard_time = time.perf_counter() - start_time

        del standard_converter

        print(f"Standard pipeline:  {standard_time:.2f} seconds")

    # Verify results
    if run_threaded and run_serial:
        assert standard_success_count == threaded_success_count
        assert standard_failure_count == threaded_failure_count
    if run_serial:
        assert standard_success_count == len(test_files)
        assert standard_failure_count == 0
    if run_threaded:
        assert threaded_success_count == len(test_files)
        assert threaded_failure_count == 0


def test_pipeline_comparison():
    """Compare all three pipeline implementations"""
    test_file = "tests/data/pdf/2206.01062.pdf"

    # Sync pipeline
    sync_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=StandardPdfPipeline,
            )
        }
    )

    start_time = time.perf_counter()
    sync_results = list(sync_converter.convert_all([test_file]))
    sync_time = time.perf_counter() - start_time

    # Threaded pipeline
    threaded_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=ThreadedStandardPdfPipeline,
                pipeline_options=ThreadedPdfPipelineOptions(
                    layout_batch_size=1,
                    ocr_batch_size=1,
                    table_batch_size=1,
                ),
            )
        }
    )

    start_time = time.perf_counter()
    threaded_results = list(threaded_converter.convert_all([test_file]))
    threaded_time = time.perf_counter() - start_time

    print("\nPipeline Comparison:")
    print(f"Sync pipeline:     {sync_time:.2f} seconds")
    print(f"Threaded pipeline: {threaded_time:.2f} seconds")
    print(f"Speedup:           {sync_time / threaded_time:.2f}x")

    # Verify results are equivalent
    assert len(sync_results) == len(threaded_results) == 1
    assert (
        sync_results[0].status == threaded_results[0].status == ConversionStatus.SUCCESS
    )

    # Basic content comparison
    sync_doc = sync_results[0].document
    threaded_doc = threaded_results[0].document

    assert len(sync_doc.pages) == len(threaded_doc.pages)
    assert len(sync_doc.texts) == len(threaded_doc.texts)


def test_pypdfium_threaded_pipeline():
    doc_converter = (
        DocumentConverter(  # all of the below is optional, has internal defaults.
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=ThreadedStandardPdfPipeline,
                    backend=PyPdfiumDocumentBackend,
                ),
            },
        )
    )

    test_file = "tests/data/pdf/2206.01062.pdf"
    for i in range(6):
        print(f"iteration {i=}")
        conv_result = doc_converter.convert(test_file)
        assert conv_result.status == ConversionStatus.SUCCESS
        print(f"[{i=}] Success")
    print("All done!")


if __name__ == "__main__":
    # Run basic performance test
    test_pipeline_comparison()
