"""ARGAvatar Gradio UI assembly: `build_ui()` and the per-tab builders.

Two tabs, both driven by `scripts/inference/pipeline.ARGAvatarPipeline`:

**Reconstruct** -- upload one or more images of the same person (one person per image).
Returns an interactive 3D Gaussian viewer (drag to change the viewing direction) and a
downloadable `.ply` of the canonical Gaussians.

**Animate** -- take that avatar and drive it with a motion: either one of the motions
bundled in `assets/example_motion/`, or a video you upload, which is first run through
`scripts/inference/estimate_motion_mhr.py` to fit MHR poses. Fitting a new motion from
video needs `kaolin` (installed by `requirements.txt`);
the bundled motions play without it.

This module is UI assembly only -- every callback lives in `app/callbacks.py` and the
per-session state in `app/state.py`. The CLI entrypoint is `app/__main__.py`.
"""
from __future__ import annotations

from typing import Final

import gradio as gr
from app.streaming import StreamingBlocks

from app.callbacks import (
    ASSETS,
    WORK_ROOT,
    bundled_motions,
    cleanup_work_dirs,
    example_subjects,
    get_motion_preview,
    make_remove_user_handler,
    make_select_handler,
    make_use_all_handler,
    on_auto_select_frames,
    on_frames_gallery_select,
    on_image_deleted,
    on_images_uploaded,
    on_input_mode_change,
    on_motion_source_change,
    on_video_cleared,
    on_video_uploaded,
    pipeline_status_line,
    run_animate,
    run_reconstruct,
    set_no_load,
    set_pipeline,
    use_input_video,
)
from app.viewer import PLACEHOLDER_HTML
from app.state import AppState, InputMode, MotionSource
from scripts.inference.pipeline import (
    DEFAULT_CONFIG_PATH,
    ARGAvatarPipeline,
    PipelineOptions,
)

# One GPU, one model: serialise reconstruction and animation jobs across sessions.
CONCURRENCY_LIMIT: Final[int] = 1

APP_CSS: Final[str] = """
#app_header {
    padding: 4px 0 0 0;
}
#app_status {
    opacity: 0.75;
    font-size: 0.9em;
}
#upload_box,
#upload_box .grid-wrap,
#upload_box .grid-wrap.fixed-height,
#upload_box .grid-container,
#frames_gallery,
#frames_gallery .grid-wrap,
#frames_gallery .grid-wrap.fixed-height,
#frames_gallery .grid-container {
    min-height: 220px;
    height: auto !important;
    max-height: none !important;
    overflow: visible !important;
    overflow-y: visible !important;
}
#upload_box .thumbnail-item,
#upload_box button.gallery-item,
#frames_gallery .thumbnail-item,
#frames_gallery button.gallery-item {
    aspect-ratio: 3 / 4;
    object-fit: contain;
}
#upload_box_video video, #upload_box_video {
    aspect-ratio: 16 / 9;
    width: 100%;
}
.splat-viewer-placeholder {
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 16px;
    opacity: 0.6;
    border: 1px dashed var(--border-color-primary);
    border-radius: 8px;
}
/* The outer workflow stacks independently of Gradio's component min-widths. */
.splat-viewer, .splat-viewer-placeholder {
    height: min(75vh, 640px);
}
#app_tabs, .workflow-column { min-width: 0; }
@media (max-width: 767px), (pointer: coarse) and (max-width: 1023px) {
    .gradio-container {
        padding-left: max(12px, env(safe-area-inset-left)) !important;
        padding-right: max(12px, env(safe-area-inset-right)) !important;
        padding-bottom: max(16px, env(safe-area-inset-bottom)) !important;
    }
    .workflow-row { flex-direction: column !important; }
    .workflow-column { width: 100% !important; min-width: 0 !important; flex: auto !important; }
    #app_header h1 { font-size: 1.5rem; line-height: 1.25; }
    #app_tabs .tab-nav { flex-wrap: nowrap; overflow-x: auto; }
    #app_tabs button:not(.thumbnail-item):not(.gallery-item) { min-height: 44px; }
    #app_tabs input, #app_tabs select, #app_tabs textarea { font-size: 16px; }
    #app_tabs .primary-action { width: 100%; }
    #upload_box .delete-button { opacity: 1; min-width: 44px; min-height: 44px; }
    .avatar-gallery .grid-container { grid-template-columns: repeat(2, minmax(0, 1fr)) !important; }
    #app_tabs video, #app_tabs img { max-width: 100%; object-fit: contain; }
    .splat-viewer, .splat-viewer-placeholder {
        height: clamp(240px, 60svh, 480px);
    }
    #motion_preview_video { height: auto !important; }
}
@media (max-width: 767px) and (orientation: landscape),
       (pointer: coarse) and (max-width: 1023px) and (orientation: landscape) {
    .splat-viewer, .splat-viewer-placeholder { height: 70svh; min-height: 180px; }
}
"""


APP_JS: Final[str] = """
(() => {
    const phoneQuery = "(max-width: 767px), (pointer: coarse) and (max-width: 1023px)";
    // Safari can retain the landscape document width when returning to portrait.
    const originalMaxWidth = document.documentElement.style.maxWidth;
    const syncViewportWidth = () => {
        document.documentElement.style.maxWidth = window.matchMedia(phoneQuery).matches
            ? window.innerWidth + "px" : originalMaxWidth;
    };
    syncViewportWidth();
    window.addEventListener("resize", syncViewportWidth);
    window.visualViewport?.addEventListener("resize", syncViewportWidth);
    // Gradio's video elements do not consistently request inline playback on iOS.
    const prepareVideos = () => {
        document.querySelectorAll("#app_tabs video").forEach(video => {
            if (!video.playsInline) video.playsInline = true;
        });
    };
    prepareVideos();
    const videoObserver = new MutationObserver(prepareVideos);
    videoObserver.observe(document.body, {
        childList: true, subtree: true, attributes: true, attributeFilter: ["playsinline"]
    });
    if (!window.matchMedia(phoneQuery).matches) return;
    // launch JS may run before the components mount. Collapse once, then leave user choice alone.
    const collapseLibrary = () => {
        const button = document.querySelector("#subject_library button.label-wrap");
        if (!button) return false;
        if (button.classList.contains("open")) button.click();
        return true;
    };
    if (!collapseLibrary()) {
        const observer = new MutationObserver(() => {
            if (collapseLibrary()) observer.disconnect();
        });
        observer.observe(document.body, { childList: true, subtree: true });
        setTimeout(() => observer.disconnect(), 10000);
    }
})();
"""


def _build_header(status_text: str) -> None:
    """Build the top header banner with title, description, and load status."""
    gr.Markdown(
        "# ARGAvatar: 3D Gaussian Avatar Reconstruction and Animation\n\n"
        "Sparse-view feed-forward 3D avatar synthesis and animation pipeline. "
        "Reconstruct an avatar on the first tab, then drive it with motion on the second tab.",
        elem_id="app_header",
    )
    gr.Markdown(status_text, elem_id="app_status")


def _build_reconstruct_tab(session_state: gr.State) -> None:
    """Build UI components and event bindings for Tab 1 (Reconstruct)."""
    with gr.Tab("1. Reconstruct", id="tab_reconstruct"):
        # Constructed here (`render=False`) so the left column's upload/auto-select event
        # wiring below can reference them, but actually rendered in the right column --
        # see the `.render()` calls there.
        max_views = gr.Slider(
            1, 6, value=6, step=1, label="Max views used",
            info="Extra selected views beyond this are dropped before reconstruction; "
                 "also sets how many frames Auto-Select / video upload pick.",
            render=False,
        )
        reconstruct_btn = gr.Button(
            "Reconstruct Avatar", variant="primary", size="lg", render=False, elem_classes="primary-action"
        )

        with gr.Row(elem_classes="workflow-row"):
            with gr.Column(scale=1, min_width=0, elem_classes="workflow-column"):
                user_subjects_state = gr.State([])
                input_mode = gr.Radio(
                    [InputMode.IMAGES.value, InputMode.VIDEO.value],
                    value=InputMode.IMAGES.value,
                    label="Input Mode",
                )
                with gr.Group(visible=True) as image_mode_group:
                    gr.Markdown(
                        "For best results, use **full-body photos** of the same person, with head "
                        "and feet fully visible. Include **at least 4 views: front, back, left, and right**."
                    )
                    image_gallery = gr.Gallery(
                        label="Upload Images / Selected Views",
                        columns=3,
                        fit_columns=False,
                        interactive=True,
                        file_types=["image"],
                        allow_preview=True,
                        preview=False,
                        object_fit="contain",
                        elem_id="upload_box",
                        elem_classes="avatar-gallery",
                    )
                    gr.Markdown(
                        "Tap to upload photos of the same person from different angles, or select "
                        "views in the **Subject Library**. Use ✕ to remove a selected photo."
                    )
                with gr.Group(visible=False) as video_mode_group:
                    video_upload = gr.Video(
                        label="Upload Video",
                        elem_id="upload_box_video",
                    )
                    auto_select_btn = gr.Button(
                        "Auto-Select Frames (LPIPS FPS)", size="sm",
                    )
                    frames_gallery = gr.Gallery(
                        label="Extracted Video Frames",
                        columns=3,
                        fit_columns=False,
                        interactive=False,
                        allow_preview=False,
                        preview=False,
                        object_fit="contain",
                        visible=False,
                        elem_id="frames_gallery",
                        elem_classes="avatar-gallery",
                    )
                    gr.Markdown(
                        "Upload a full-body turning video (**20 MB maximum**, MP4 or MOV). "
                        "Frames are selected automatically; tap a frame to remove it. "
                        "**Auto-Select Frames** picks diverse views. You can reuse the video in Animate."
                    )
                example_subs = example_subjects("example_images")
                with gr.Accordion("Subject Library", open=True, elem_id="subject_library"):
                    @gr.render(inputs=[user_subjects_state])
                    def render_subject_library(user_subs: list[dict]) -> None:
                        with gr.Tabs():
                            for folder_name, img_paths in example_subs.items():
                                with gr.Tab(folder_name):
                                    use_all_btn = gr.Button(
                                        f"Use all {len(img_paths)} views",
                                        size="sm",
                                    )
                                    preset_gallery = gr.Gallery(
                                        value=img_paths,
                                        elem_classes="avatar-gallery",
                                        columns=3,
                                        fit_columns=False,
                                        show_label=False,
                                        allow_preview=False,
                                        preview=False,
                                        object_fit="contain",
                                    )
                                    preset_gallery.select(
                                        make_select_handler(img_paths, associated_video=None),
                                        inputs=[image_gallery, session_state],
                                        outputs=[image_gallery, input_mode, image_mode_group, video_mode_group, session_state],
                                    )
                                    use_all_btn.click(
                                        make_use_all_handler(img_paths, associated_video=None),
                                        inputs=[session_state],
                                        outputs=[image_gallery, input_mode, image_mode_group, video_mode_group, session_state],
                                    )

                            for sub in (user_subs or []):
                                sub_id = sub["id"]
                                sub_name = sub["name"]
                                sub_paths = sub["paths"]
                                sub_video = sub.get("video_path")
                                with gr.Tab(sub_name):
                                    with gr.Row():
                                        use_user_btn = gr.Button(
                                            f"Use all {len(sub_paths)} views",
                                            size="sm",
                                        )
                                        remove_user_btn = gr.Button(
                                            f"Remove {sub_name}",
                                            size="sm",
                                            variant="stop",
                                        )
                                    user_gallery = gr.Gallery(
                                        value=sub_paths,
                                        elem_classes="avatar-gallery",
                                        columns=3,
                                        fit_columns=False,
                                        show_label=False,
                                        allow_preview=False,
                                        preview=False,
                                        object_fit="contain",
                                    )
                                    user_gallery.select(
                                        make_select_handler(sub_paths, associated_video=sub_video),
                                        inputs=[image_gallery, session_state],
                                        outputs=[image_gallery, input_mode, image_mode_group, video_mode_group, session_state],
                                    )
                                    use_user_btn.click(
                                        make_use_all_handler(sub_paths, associated_video=sub_video),
                                        inputs=[session_state],
                                        outputs=[image_gallery, input_mode, image_mode_group, video_mode_group, session_state],
                                    )
                                    remove_user_btn.click(
                                        make_remove_user_handler(sub_id),
                                        inputs=[user_subjects_state],
                                        outputs=[user_subjects_state],
                                    )

                input_mode.change(
                    on_input_mode_change,
                    inputs=[input_mode],
                    outputs=[image_mode_group, video_mode_group],
                )
                image_gallery.upload(
                    on_images_uploaded,
                    inputs=[image_gallery, user_subjects_state, session_state],
                    outputs=[image_gallery, user_subjects_state, session_state],
                )
                image_gallery.delete(
                    on_image_deleted,
                    inputs=[image_gallery, session_state],
                    outputs=[session_state],
                )
                video_upload.upload(
                    on_video_uploaded,
                    inputs=[video_upload, user_subjects_state, max_views, session_state],
                    outputs=[frames_gallery, user_subjects_state, session_state],
                )
                video_upload.clear(
                    on_video_cleared,
                    inputs=[session_state],
                    outputs=[frames_gallery, session_state],
                )
                frames_gallery.select(
                    on_frames_gallery_select,
                    inputs=[frames_gallery, session_state],
                    outputs=[frames_gallery, session_state],
                )
                auto_select_btn.click(
                    on_auto_select_frames,
                    inputs=[session_state, max_views],
                    outputs=[frames_gallery, session_state],
                )

            with gr.Column(scale=1, min_width=0, elem_classes="workflow-column"):
                with gr.Accordion("Advanced Parameters", open=False):
                    max_views.render()
                reconstruct_btn.render()
                status_out = gr.Markdown()
                viewer_out = gr.HTML(PLACEHOLDER_HTML, label="3D Avatar Viewer", elem_id="viewer_out")
                ply_out = gr.DownloadButton(
                    "Download Canonical Gaussians (.ply)",
                    variant="secondary",
                    visible=False,
                )

        reconstruct_btn.click(
            run_reconstruct,
            inputs=[
                input_mode, image_gallery, video_upload, frames_gallery,
                max_views, session_state,
            ],
            outputs=[viewer_out, ply_out, status_out, session_state],
            concurrency_limit=CONCURRENCY_LIMIT,
            concurrency_id="avatar_gpu",
        )


def _build_animate_tab(session_state: gr.State) -> None:
    """Build UI components and event bindings for Tab 2 (Animate)."""
    motions = bundled_motions()
    motion_names = list(motions)

    with gr.Tab("2. Animate", id="tab_animate"):
        gr.Markdown(
            "Drive the avatar you reconstructed with preset motions or an uploaded driving video. "
            "Select a preset or upload a full-body motion video."
        )
        with gr.Row(elem_classes="workflow-row"):
            with gr.Column(scale=4, min_width=0, elem_classes="workflow-column"):
                motion_source = gr.Radio(
                    choices=[MotionSource.BUNDLED.value, MotionSource.CUSTOM.value],
                    value=MotionSource.BUNDLED.value,
                    label="Motion Source",
                )
                use_input_video_btn = gr.Button(
                    "Use Input Video", variant="secondary"
                )
                with gr.Group(visible=True) as preset_group:
                    motion_in = gr.Dropdown(
                        motion_names,
                        value=motion_names[0] if motion_names else None,
                        label="Preset Motion Library",
                    )
                    motion_preview_out = gr.Video(
                        value=get_motion_preview(motion_names[0]) if motion_names else None,
                        label="Preset Motion Preview (Posed Mesh)",
                        interactive=False,
                        height=240,
                        elem_id="motion_preview_video",
                    )
                with gr.Group(visible=False) as custom_group:
                    gr.Markdown("Upload a driving video (**20 MB maximum**, MP4 or MOV).")
                    video_in = gr.Video(label="Upload Driving Video")
                    example_videos = (
                        sorted((ASSETS / "example_videos").glob("*.mp4"))
                        if (ASSETS / "example_videos").is_dir()
                        else []
                    )
                    if example_videos:
                        gr.Examples(
                            examples=[[str(p)] for p in example_videos],
                            inputs=[video_in],
                            label="Example Driving Videos",
                        )
                frames_in = gr.Slider(
                    0, 300, value=120, step=10,
                    label="Max frames (0 = all)",
                )
                animate_btn = gr.Button("Render Animation", variant="primary", size="lg", elem_classes="primary-action")

                motion_source.change(
                    on_motion_source_change,
                    inputs=[motion_source],
                    outputs=[preset_group, custom_group],
                )
                motion_in.change(
                    get_motion_preview,
                    inputs=[motion_in],
                    outputs=[motion_preview_out],
                )

            with gr.Column(scale=6, min_width=0, elem_classes="workflow-column"):
                animate_status = gr.Markdown(
                    "Reconstruct an avatar on **1. Reconstruct** first — this tab drives that avatar."
                )
                live_out = gr.Image(
                    label="Rendering (live)", visible=False, interactive=False,
                    format="png",  # Rendered frames can be RGBA; JPEG cannot encode alpha.
                )
                original_out = gr.Video(
                    label="Rendered Animation", streaming=True, autoplay=True,
                    elem_id="rendered_motion_video",
                )

        use_input_video_btn.click(
            use_input_video,
            inputs=[session_state],
            outputs=[motion_source, video_in, preset_group, custom_group, animate_status],
        )

        animate_btn.click(
            run_animate,
            inputs=[motion_source, motion_in, video_in, frames_in, session_state],
            outputs=[live_out, original_out, animate_status, session_state],
            concurrency_limit=CONCURRENCY_LIMIT,
            concurrency_id="avatar_gpu",
        )


def build_ui(status_text: str = "") -> gr.Blocks:
    """Construct the top-level Gradio Blocks application."""
    with StreamingBlocks(title="ARGAvatar") as ui:
        session_state = gr.State(AppState())
        _build_header(status_text or pipeline_status_line())
        with gr.Tabs(elem_id="app_tabs"):
            _build_reconstruct_tab(session_state)
            _build_animate_tab(session_state)

    return ui
