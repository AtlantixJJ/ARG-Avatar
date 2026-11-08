"""Keep Gradio's session component values consistent with streamed media outputs."""
import gradio as gr
from gradio.components.base import StreamingOutput
from gradio.utils import is_prop_update
from pathlib import Path
from app.files import COOKIE_NAME, SessionFiles, _active_owner


class StreamingBlocks(gr.Blocks):
    def __init__(self, *args, **kwargs):
        self.file_access = SessionFiles(Path("assets"))
        super().__init__(*args, **kwargs)

    async def process_api(self, block_fn, inputs, state=None, request=None, **kwargs):
        # Queue callbacks run outside the originating HTTP context, so explicitly
        # carry the request's browser identity into worker threads and generators.
        owner = request.cookies.get(COOKIE_NAME) if request is not None else None
        token = None
        if owner:
            session = kwargs.get("session_hash") or request.session_hash
            if session and not self.file_access.bind_session(session, owner):
                raise gr.Error("This session belongs to another browser.")
            self.file_access.validate_inputs(inputs, owner)
            token = _active_owner.set((self.file_access, owner))
        try:
            return await super().process_api(
                block_fn, inputs, state=state, request=request, **kwargs
            )
        finally:
            if token is not None:
                _active_owner.reset(token)

    async def postprocess_data(self, block_fn, predictions, state):
        active = _active_owner.get()
        if active:
            access, owner = active
            access.register(predictions, owner)
        result = await super().postprocess_data(block_fn, predictions, state)
        if active:
            access.register(result, owner)
        return result

    async def handle_streaming_outputs(self, block_fn, data, session_hash, run,
                                       root_path=None, final=False):
        existing = self.pending_streams.get(session_hash, {}).get(run, {})
        started = set(existing)
        data = await super().handle_streaming_outputs(
            block_fn, data, session_hash, run, root_path, final
        )
        if session_hash is not None and run is not None:
            state = self.state_holder[session_hash]
            for index, (block, value) in enumerate(zip(block_fn.outputs, data)):
                if isinstance(block, StreamingOutput) and block.streaming and not is_prop_update(value):
                    # Gradio 6 Video uses flat FileData, but stream_output still
                    # returns the older VideoData {"video": FileData} shape.
                    if (isinstance(block, gr.Video) and block.data_model is gr.FileData
                            and isinstance(value, dict) and "video" in value):
                        value = data[index] = value["video"]
                    state._update_value_in_config(block._id, value)
                    if block._id in started:
                        # The playlist URL is stable and updates independently.
                        # Reassigning it per chunk makes the Gradio 6 player attach
                        # another HLS instance to the same video element.
                        data[index] = gr.skip()
        return data
