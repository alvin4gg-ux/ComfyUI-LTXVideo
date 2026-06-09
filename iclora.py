import logging
import comfy
import comfy.samplers
import comfy.utils
import comfy.sd
import comfy_extras.nodes_lt as nodes_lt
import folder_paths
import node_helpers
import torch
from comfy_api.latest import io
from .latents import LTXVDilateLatent
from .nodes_registry import NODES_DISPLAY_NAME_PREFIX, comfy_node


@comfy_node(name="LTXAddVideoICLoRAGuide")
class LTXAddVideoICLoRAGuide(io.ComfyNode):
    PATCHIFIER = nodes_lt.SymmetricPatchifier(1, start_end=True)
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXAddVideoICLoRAGuide",
            display_name=NODES_DISPLAY_NAME_PREFIX + " Add Video IC-LoRA Guide",
            category="Lightricks/IC-LoRA",
            description=(
                "Adds one or more conditioning frames starting at the specified frame index. "
                "Supports both single images and multi-frame videos. "
                "The latent_downscale_factor resizes input to a fraction of the target size "
                "(1 = original, 2 = half, 3 = third, etc.) for IC-LoRA on small grids."
            ),
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Latent.Input(
                    "latent",
                    tooltip="Video-only latent to condition. Must be a 5D video latent (batch, channels, frames, height, width).",
                ),
                io.Image.Input("image"),
                io.Int.Input(
                    "frame_idx",
                    default=0,
                    min=-9999,
                    max=9999,
                    tooltip=(
                        "Frame index to start the conditioning at. "
                        "For single-frame videos, any frame_idx value is acceptable. "
                        "For videos, frame_idx must be 1 modulo 8, otherwise it will be rounded "
                        "down to the nearest 1 modulo 8. Negative values are counted from the end of the video."
                    ),
                ),
                io.Float.Input(
                    "strength",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                ),
                io.Float.Input(
                    "latent_downscale_factor",
                    default=1.0,
                    min=1.0,
                    max=10.0,
                    step=1.0,
                    tooltip="For IC-LoRA on small grid. 1 means original size, 2 means half size, 3 means third, etc.",
                ),
                io.Combo.Input(
                    "crop",
                    options=["disabled", "center"],
                    default="disabled",
                    tooltip="Crop mode when resizing. 'center' crops to fit, 'disabled' stretches to fit.",
                ),
                io.Boolean.Input(
                    "use_tiled_encode",
                    default=False,
                    tooltip="Enable tiled VAE encoding for large resolutions/long videos to reduce memory usage.",
                ),
                io.Int.Input(
                    "tile_size",
                    default=256,
                    min=64,
                    max=512,
                    step=32,
                    tooltip="Spatial tile size for tiled encoding. Only used when use_tiled_encode is enabled.",
                ),
                io.Int.Input(
                    "tile_overlap",
                    default=64,
                    min=16,
                    max=256,
                    step=16,
                    tooltip="Overlap between tiles for tiled encoding. Only used when use_tiled_encode is enabled.",
                ),
            ],
            outputs=[
                io.Conditioning.Output("positive"),
                io.Conditioning.Output("negative"),
                io.Latent.Output("latent"),
            ],
        )
    @classmethod
    def encode(
        cls,
        vae,
        latent_width,
        latent_height,
        images,
        scale_factors,
        latent_downscale_factor,
        crop,
        use_tiled_encode,
        tile_size,
        tile_overlap,
    ):
        time_scale_factor, width_scale_factor, height_scale_factor = scale_factors
        num_frames_to_keep = ((images.shape[0] - 1) // time_scale_factor) * time_scale_factor + 1
        images = images[:num_frames_to_keep]
        target_width = int(latent_width * width_scale_factor / latent_downscale_factor)
        target_height = int(latent_height * height_scale_factor / latent_downscale_factor)
        pixels = comfy.utils.common_upscale(images.movedim(-1, 1), target_width, target_height, "bilinear", crop=crop).movedim(1, -1)
        encode_pixels = pixels[:, :, :, :3]
        if use_tiled_encode:
            guide_latent = vae.encode_tiled(encode_pixels, tile_x=tile_size, tile_y=tile_size, overlap=tile_overlap)
        else:
            guide_latent = vae.encode(encode_pixels)
        return encode_pixels, guide_latent
    @classmethod
    def execute(
        cls,
        positive,
        negative,
        vae,
        latent,
        image,
        frame_idx,
        strength,
        latent_downscale_factor,
        crop,
        use_tiled_encode,
        tile_size,
        tile_overlap,
    ) -> io.NodeOutput:
        scale_factors = vae.downscale_index_formula
        latent_image = latent["samples"].clone()
        noise_mask = nodes_lt.get_noise_mask(latent).clone()
        _, _, latent_length, latent_height, latent_width = latent_image.shape
        time_scale_factor = scale_factors[0]
        num_frames_to_keep = ((image.shape[0] - 1) // time_scale_factor) * time_scale_factor + 1
        causal_fix = frame_idx == 0 or num_frames_to_keep == 1
        if not causal_fix:
            image = torch.cat([image[:1], image], dim=0)
        image, guide_latent = cls.encode(vae, latent_width, latent_height, image, scale_factors, latent_downscale_factor, crop, use_tiled_encode, tile_size, tile_overlap)
        if not causal_fix:
            guide_latent = guide_latent[:, :, 1:, :, :]
            image = image[1:]
        guide_orig_shape = list(guide_latent.shape[2:])
        guide_mask = None
        if latent_downscale_factor > 1:
            if latent_width % latent_downscale_factor != 0 or latent_height % latent_downscale_factor != 0:
                raise ValueError(f"Latent spatial size {latent_width}x{latent_height} must be divisible by latent_downscale_factor {latent_downscale_factor}")
            dilated = LTXVDilateLatent().dilate_latent({"samples": guide_latent}, horizontal_scale=int(latent_downscale_factor), vertical_scale=int(latent_downscale_factor))[0]
            guide_mask = dilated["noise_mask"]
            guide_latent = dilated["samples"]
        iclora_tokens_added = guide_latent.shape[2] * guide_latent.shape[3] * guide_latent.shape[4]
        frame_idx, latent_idx = nodes_lt.LTXVAddGuide.get_latent_index(positive, latent_length, len(image), frame_idx, scale_factors)
        assert latent_idx + guide_latent.shape[2] <= latent_length, "Conditioning frames exceed the length of the latent sequence."
        # 临时纯latent字典，规避AV检测
        tmp_lat_dict = {"samples": latent_image, "noise_mask": noise_mask}
        positive, negative, tmp_out, _ = nodes_lt.LTXVAddGuide.append_keyframe(
            positive,
            negative,
            frame_idx,
            tmp_lat_dict["samples"],
            tmp_lat_dict["noise_mask"],
            guide_latent,
            strength,
            scale_factors,
            guide_mask=guide_mask,
            latent_downscale_factor=latent_downscale_factor,
            causal_fix=causal_fix,
        )
        latent_image = tmp_out
        from .iclora_attention import append_guide_attention_entry
        positive = append_guide_attention_entry(positive, iclora_tokens_added, guide_orig_shape)
        negative = append_guide_attention_entry(negative, iclora_tokens_added, guide_orig_shape)
        out_lat = latent.copy()
        out_lat["samples"] = latent_image
        out_lat["noise_mask"] = noise_mask
        return io.NodeOutput(positive, negative, out_lat)


@comfy_node(name="LTXAddVideoICLoRAGuideAdvanced")
class LTXAddVideoICLoRAGuideAdvanced(LTXAddVideoICLoRAGuide):
    @classmethod
    def define_schema(cls):
        input_list = [
            io.Conditioning.Input("positive"),
            io.Conditioning.Input("negative"),
            io.Vae.Input("vae"),
            io.Latent.Input("latent", tooltip="支持AV混合latent，内部自动剥离纯视频"),
            io.Image.Input("ref1", optional=True, tooltip="主体参考图1"),
            io.String.Input("desc1", default="", multiline=True, tooltip="参考1备注"),
            io.Image.Input("ref2", optional=True, tooltip="主体参考图2"),
            io.String.Input("desc2", default="", multiline=True),
            io.Image.Input("ref3", optional=True, tooltip="主体参考图3"),
            io.String.Input("desc3", default="", multiline=True),
            io.Image.Input("ref4", optional=True, tooltip="主体参考图4"),
            io.String.Input("desc4", default="", multiline=True),
            io.Image.Input("bg1", optional=True, tooltip="背景参考1"),
            io.Boolean.Input("enable_bg1", default=True),
            io.String.Input("desc_bg1", default="", multiline=True),
            io.Image.Input("bg2", optional=True),
            io.Boolean.Input("enable_bg2", default=False),
            io.String.Input("desc_bg2", default="", multiline=True),
            io.Image.Input("bg3", optional=True),
            io.Boolean.Input("enable_bg3", default=False),
            io.String.Input("desc_bg3", default="", multiline=True),
            io.Image.Input("bg4", optional=True),
            io.Boolean.Input("enable_bg4", default=False),
            io.String.Input("desc_bg4", default="", multiline=True),
            io.Image.Input("bg5", optional=True),
            io.Boolean.Input("enable_bg5", default=False),
            io.String.Input("desc_bg5", default="", multiline=True),
            io.Image.Input("bg6", optional=True),
            io.Boolean.Input("enable_bg6", default=False),
            io.String.Input("desc_bg6", default="", multiline=True),
            io.Float.Input("strength", default=1.0, min=0.0, max=1.0, step=0.01),
            io.Float.Input("latent_downscale_factor", default=1.0, min=1.0, max=10.0, step=1.0),
            io.Combo.Input("crop", options=["disabled", "center"], default="disabled"),
            io.Boolean.Input("use_tiled_encode", default=False),
            io.Int.Input("tile_size", default=256, min=64, max=512, step=32),
            io.Int.Input("tile_overlap", default=64, min=16, max=256, step=16),
            io.Float.Input("attention_strength", default=1.0, min=0.0, max=1.0, step=0.01),
            io.Mask.Input("attention_mask", optional=True),
        ]
        return io.Schema(
            node_id="LTXAddVideoICLoRAGuideAdvanced",
            display_name=NODES_DISPLAY_NAME_PREFIX + " Add Video IC-LoRA Guide Advanced【多图分段·兼容AV】",
            category="Lightricks/IC-LoRA",
            description="4主体+6背景多图IC，自动均分帧，原生兼容AV音视频混合latent",
            inputs=input_list,
            outputs=[io.Conditioning.Output("positive"),io.Conditioning.Output("negative"),io.Latent.Output("latent")],
        )

    @classmethod
    def execute(
        cls, positive, negative, vae, latent,
        desc1="", desc2="", desc3="", desc4="",
        ref1=None, ref2=None, ref3=None, ref4=None,
        bg1=None, enable_bg1=True, desc_bg1="",
        bg2=None, enable_bg2=False, desc_bg2="",
        bg3=None, enable_bg3=False, desc_bg3="",
        bg4=None, enable_bg4=False, desc_bg4="",
        bg5=None, enable_bg5=False, desc_bg5="",
        bg6=None, enable_bg6=False, desc_bg6="",
        strength=1.0, latent_downscale_factor=1.0, crop="disabled",
        use_tiled_encode=False, tile_size=256, tile_overlap=64,
        attention_strength=1.0, attention_mask=None,
    ) -> io.NodeOutput:
        from .iclora_attention import normalize_mask, append_guide_attention_entry
        scale_factors = vae.downscale_index_formula
        latent_image = latent["samples"].clone()
        noise_mask = nodes_lt.get_noise_mask(latent).clone()
        _, _, total_frames, latent_h, latent_w = latent_image.shape
        pos_cond = [c.copy() for c in positive]
        neg_cond = [c.copy() for c in negative]

        ref_list = []
        if ref1 is not None: ref_list.append({"img": ref1, "text": desc1.strip()})
        if ref2 is not None: ref_list.append({"img": ref2, "text": desc2.strip()})
        if ref3 is not None: ref_list.append({"img": ref3, "text": desc3.strip()})
        if ref4 is not None: ref_list.append({"img": ref4, "text": desc4.strip()})
        if enable_bg1 and bg1 is not None: ref_list.append({"img": bg1, "text": desc_bg1.strip()})
        if enable_bg2 and bg2 is not None: ref_list.append({"img": bg2, "text": desc_bg2.strip()})
        if enable_bg3 and bg3 is not None: ref_list.append({"img": bg3, "text": desc_bg3.strip()})
        if enable_bg4 and bg4 is not None: ref_list.append({"img": bg4, "text": desc_bg4.strip()})
        if enable_bg5 and bg5 is not None: ref_list.append({"img": bg5, "text": desc_bg5.strip()})
        if enable_bg6 and bg6 is not None: ref_list.append({"img": bg6, "text": desc_bg6.strip()})

        if len(ref_list) == 0:
            out_lat = latent.copy()
            out_lat["samples"] = latent_image
            out_lat["noise_mask"] = noise_mask
            return io.NodeOutput(pos_cond, neg_cond, out_lat)

        seg_len = total_frames // len(ref_list)
        remain = total_frames % len(ref_list)
        current_start = 0
        norm_mask = normalize_mask(attention_mask)

        for idx, item in enumerate(ref_list):
            img_tensor = item["img"]
            seg_frames = seg_len + (1 if idx < remain else 0)
            frame_idx = current_start
            time_scale = scale_factors[0]
            keep_num = ((img_tensor.shape[0]-1)//time_scale)*time_scale +1
            img_in = img_tensor[:keep_num]
            causal_fix = frame_idx == 0 or keep_num ==1
            if not causal_fix:
                img_in = torch.cat([img_in[:1], img_in], dim=0)

            _, guide_lat = cls.encode(vae, latent_w, latent_h, img_in, scale_factors, latent_downscale_factor, crop, use_tiled_encode, tile_size, tile_overlap)
            if not causal_fix:
                guide_lat = guide_lat[:, :, 1:, :, :]

            guide_mask = None
            if latent_downscale_factor > 1:
                if latent_w % latent_downscale_factor !=0 or latent_h % latent_downscale_factor !=0:
                    raise ValueError(f"尺寸无法被缩放系数{latent_downscale_factor}整除")
                dilated = LTXVDilateLatent().dilate_latent({"samples":guide_lat},int(latent_downscale_factor),int(latent_downscale_factor))[0]
                guide_mask = dilated["noise_mask"]
                guide_lat = dilated["samples"]

            _, latent_idx = nodes_lt.LTXVAddGuide.get_latent_index(pos_cond, total_frames, seg_frames, frame_idx, scale_factors)
            assert latent_idx + guide_lat.shape[2] <= total_frames, "参考帧超出视频总长度"

            # 【核心修复】只用临时纯视频字典，永远不传入AV混合完整latent
            tmp_samp = latent_image.clone()
            tmp_noise = noise_mask.clone()
            pos_cond, neg_cond, new_samp, new_noise = nodes_lt.LTXVAddGuide.append_keyframe(
                pos_cond, neg_cond, frame_idx, tmp_samp, tmp_noise,
                guide_lat, strength, scale_factors, guide_mask=guide_mask,
                latent_downscale_factor=latent_downscale_factor, causal_fix=causal_fix
            )
            latent_image = new_samp
            noise_mask = new_noise

            token_num = guide_lat.shape[2]*guide_lat.shape[3]*guide_lat.shape[4]
            orig_shape = list(guide_lat.shape[2:])
            pos_cond = append_guide_attention_entry(pos_cond, token_num, orig_shape, attention_strength=attention_strength, attention_mask=norm_mask)
            neg_cond = append_guide_attention_entry(neg_cond, token_num, orig_shape, attention_strength=attention_strength, attention_mask=norm_mask)

            current_start += seg_frames

        out_lat = latent.copy()
        out_lat["samples"] = latent_image
        out_lat["noise_mask"] = noise_mask
        return io.NodeOutput(pos_cond, neg_cond, out_lat)


@comfy_node(name="LTXICLoRALoaderModelOnly")
class LTXICLoRALoaderModelOnly(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXICLoRALoaderModelOnly",
            display_name=NODES_DISPLAY_NAME_PREFIX + " IC-LoRA Loader Model Only",
            category="Lightricks/IC-LoRA",
            inputs=[io.Model.Input("model"),io.Combo.Input("lora_name",options=folder_paths.get_filename_list("loras")),io.Float.Input("strength_model",default=1.0,min=-100.0,max=100.0,step=0.01)],
            outputs=[io.Model.Output("model"),io.Float.Output("latent_downscale_factor")],
        )
    @classmethod
    def execute(cls, model, lora_name, strength_model) -> io.NodeOutput:
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora, metadata = comfy.utils.load_torch_file(lora_path, safe_load=True, return_metadata=True)
        try:
            latent_downscale_factor = float(metadata["reference_downscale_factor"])
        except (KeyError, ValueError, TypeError):
            latent_downscale_factor = 1.0
            logging.warning("Failed to extract reference_downscale_factor from metadata for %s, using 1.0",lora_path)
        if strength_model == 0:
            return io.NodeOutput(model, latent_downscale_factor)
        model_lora, _ = comfy.sd.load_lora_for_models(model, None, lora, strength_model, 0)
        return io.NodeOutput(model_lora, latent_downscale_factor)


def _patchify_audio_latent(latent):
    b, c, t, f = latent.shape
    ref_tokens = latent.permute(0, 2, 1, 3).reshape(b, t, c * f)
    return {"tokens": ref_tokens}


@comfy_node(name="LTXVSetAudioRefTokens")
class LTXVSetAudioRefTokens(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXVSetAudioRefTokens",
            display_name=NODES_DISPLAY_NAME_PREFIX + " Set Audio Ref Tokens",
            category="Lightricks/IC-LoRA",
            inputs=[io.Conditioning.Input("positive"),io.Conditioning.Input("negative"),io.Latent.Input("audio_latent")],
            outputs=[io.Conditioning.Output("positive"),io.Conditioning.Output("negative"),io.Latent.Output("frozen_audio")],
        )
    @classmethod
    def execute(cls, positive, negative, audio_latent) -> io.NodeOutput:
        latent = audio_latent["samples"]
        ref_audio = _patchify_audio_latent(latent)
        positive = node_helpers.conditioning_set_values(positive, {"ref_audio": ref_audio})
        negative = node_helpers.conditioning_set_values(negative, {"ref_audio": ref_audio})
        frozen = audio_latent.copy()
        b, c, t, f = latent.shape
        frozen["noise_mask"] = torch.zeros((b, 1, t, 1), dtype=torch.float32)
        return io.NodeOutput(positive, negative, frozen)
