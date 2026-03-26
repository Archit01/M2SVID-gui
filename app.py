import torch
import gradio as gr
import os
import subprocess
import glob
import re
import shutil
import json
import sys
import platform

def browse_folder(current_val):
    """Open a native Windows folder picker dialog using PowerShell (no tkinter needed)."""
    if platform.system() != "Windows":
        return current_val
    try:
        result = subprocess.run(
            [
                "powershell", "-NoProfile", "-Command",
                "Add-Type -AssemblyName System.Windows.Forms;"
                "$dlg = New-Object System.Windows.Forms.FolderBrowserDialog;"
                "$dlg.Description = 'Select Folder';"
                "$dlg.ShowNewFolderButton = $true;"
                "if ($dlg.ShowDialog() -eq 'OK') { $dlg.SelectedPath } else { '' }"
            ],
            capture_output=True, text=True, timeout=120
        )
        folder_path = result.stdout.strip()
        if folder_path:
            return folder_path
    except Exception:
        pass
    return current_val

# Paths configuration
env_vars = os.environ.copy()
env_vars['PYTHONPATH'] = f".;.\\third_party\\Hi3D-Official;.\\third_party\\pytorch-msssim;{env_vars.get('PYTHONPATH', '')}"

def parse_res(res_str):
    w, h = map(int, res_str.lower().split('x'))
    return w, h

def run_subprocess_with_progress(cmd, env, progress_desc="Processing"):
    process = subprocess.Popen(
        cmd, env=env, 
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
        universal_newlines=True, errors='replace',
        bufsize=1
    )
    
    percent_re = re.compile(r'(\d+)%\|')
    
    buffer = ""
    full_log = []
    while True:
        char = process.stdout.read(1)
        if not char and process.poll() is not None:
            break
        if char:
            if char in ['\r', '\n']:
                if buffer:
                    full_log.append(buffer)
                    match = percent_re.search(buffer)
                    if match:
                        perc = int(match.group(1))
                        desc = progress_desc
                        if ":" in buffer:
                            desc_part = buffer.split(":")[0].strip()
                            desc_part = re.sub(r'[^a-zA-Z0-9\s-]', '', desc_part)
                            desc = f"{progress_desc} - {desc_part}"
                        yield perc, desc
                buffer = ""
            else:
                buffer += char
                
    process.communicate()
    if process.returncode != 0:
        error_msg = "\n".join(full_log[-20:]) # Get last 20 lines of log
        raise Exception(f"Command failed with code {process.returncode}:\n{error_msg}")

def process_warping(
    input_folder, depth_folder, left_eye_folder, high_res_folder, low_res_folder,
    disparity_perc, high_batch, high_res, enable_low_res, low_batch, low_res
):
    if not input_folder or not os.path.isdir(input_folder):
        yield 0, 0, "Input folder does not exist.", "Error"
        return
    if not depth_folder or not os.path.isdir(depth_folder):
        yield 0, 0, "Depth folder does not exist.", "Error"
        return
    if not left_eye_folder or not high_res_folder or (enable_low_res and not low_res_folder):
        yield 0, 0, "Please define all required output folders.", "Error"
        return

    os.makedirs(left_eye_folder, exist_ok=True)
    os.makedirs(high_res_folder, exist_ok=True)
    if enable_low_res:
        os.makedirs(low_res_folder, exist_ok=True)
        target_res_str = low_res
    else:
        target_res_str = high_res

    w_target, h_target = parse_res(target_res_str)
    w_high, h_high = parse_res(high_res)

    video_files = glob.glob(os.path.join(input_folder, "*.mp4"))
    
    total_files = len(video_files)
    if total_files == 0:
        yield 0, 0, "No mp4 files found in input folder.", "Error"
        return

    for i, video_path in enumerate(video_files):
        file_perc = int((i / total_files) * 100)
        filename = os.path.basename(video_path)
        base_name = os.path.splitext(filename)[0]
        
        depth_path = os.path.join(depth_folder, f"{base_name}_depth.mp4")
        if not os.path.exists(depth_path):
            yield file_perc, 0, f"Skipping {filename} (no depth file)", "Running"
            continue
        
        yield file_perc, 0, f"Processing {filename}...", "Running"

        # 1. Downscale Left Eye
        left_eye_out = os.path.join(left_eye_folder, f"{base_name}_lefteye.mp4")
        cmd_scale = [
            "ffmpeg", "-y", "-i", video_path, 
            "-vf", f"scale={w_target}:{h_target},setsar=1:1", 
            "-c:v", "libx264", "-crf", "14", "-preset", "slow", "-profile:v", "high10", "-pix_fmt", "yuv420p10le",
            left_eye_out
        ]
        
        yield file_perc, 50, f"{filename} - Waiting on FFmpeg (Downscaling Left Eye)", "Running"
        res = subprocess.run(cmd_scale, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode != 0:
            raise Exception(f"FFmpeg resizing failed:\n{res.stderr.decode('utf-8')}")

        if not enable_low_res:
            high_res_in_path = left_eye_out
            temp_high_res = None
        else:
            temp_high_res = os.path.join(high_res_folder, f"{base_name}_temp_high.mp4")
            cmd_scale_high = [
                "ffmpeg", "-y", "-i", video_path, 
                "-vf", f"scale={w_high}:{h_high},setsar=1:1", 
                "-c:v", "libx264", "-crf", "14", "-preset", "slow", "-profile:v", "high10", "-pix_fmt", "yuv420p10le",
                temp_high_res
            ]
            yield file_perc, 75, f"{filename} - Waiting on FFmpeg (Scaling for High Res Warping)", "Running"
            res = subprocess.run(cmd_scale_high, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode != 0:
                raise Exception(f"FFmpeg resizing (High Res) failed:\n{res.stderr.decode('utf-8')}")
            high_res_in_path = temp_high_res

        # 2. High Res Warping (on scaled video)
        high_res_out = os.path.join(high_res_folder, f"{base_name}_{w_high}_splatted2.mp4")
        cmd_warp_high = [
            sys.executable, "warping.py",
            "--video_path", high_res_in_path,
            "--depth_path", depth_path,
            "--output_path", high_res_out,
            "--disparity_perc", str(disparity_perc),
            "--batch_size", str(high_batch),
            "--crf", "14",
            "--bit_depth", "10"
        ]
        for sub_perc, desc in run_subprocess_with_progress(cmd_warp_high, env_vars, f"High Res Warping"):
            yield file_perc, sub_perc, f"File {i+1}/{total_files} | {filename} - {desc}", "Running"

        if temp_high_res and os.path.exists(temp_high_res):
            os.remove(temp_high_res)

        # 3. Low Res Warping (Optional, on downscaled video)
        if enable_low_res:
            w_low, h_low = parse_res(low_res)
            low_res_out = os.path.join(low_res_folder, f"{base_name}_{w_low}_splatted2.mp4")
            cmd_warp_low = [
                sys.executable, "warping.py",
                "--video_path", left_eye_out,
                "--depth_path", depth_path,
                "--output_path", low_res_out,
                "--disparity_perc", str(disparity_perc),
                "--batch_size", str(low_batch),
                "--crf", "14",
                "--bit_depth", "10"
            ]
            for sub_perc, desc in run_subprocess_with_progress(cmd_warp_low, env_vars, f"Low Res Warping"):
                yield file_perc, sub_perc, f"File {i+1}/{total_files} | {filename} - {desc}", "Running"

        # 4. Move to Finish
        finish_dir = os.path.join(input_folder, "finish")
        os.makedirs(finish_dir, exist_ok=True)
        shutil.move(video_path, os.path.join(finish_dir, filename))

    yield 100, 100, "All files processed.", "Warping Section Processing Complete!"

def process_inpainting(
    left_eye_folder, grid_folder, output_folder,
    mask_antialias, tile_size, tile_overlap, chunk_size, overlap, original_input_blend_strength,
    model_variant
):
    if not left_eye_folder or not os.path.isdir(left_eye_folder):
        yield 0, 0, 0, "Left Eye folder does not exist.", "Error"
        return
    if not grid_folder or not os.path.isdir(grid_folder):
        yield 0, 0, 0, "Grid folder does not exist.", "Error"
        return
    if not output_folder:
        yield 0, 0, 0, "Please provide an output folder.", "Error"
        return
        
    os.makedirs(output_folder, exist_ok=True)
    
    inpaint_env = env_vars.copy()
    inpaint_env['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    left_eye_files = glob.glob(os.path.join(left_eye_folder, "*_lefteye.mp4"))
    total_files = len(left_eye_files)
    
    if total_files == 0:
        yield 0, 0, 0, "No *_lefteye.mp4 files found in Left Eye Folder.", "Error"
        return

    for i, left_eye_path in enumerate(left_eye_files):
        file_perc = int((i / total_files) * 100)
        filename = os.path.basename(left_eye_path)
        base_name = filename.replace("_lefteye.mp4", "")
        
        # Discover grid video containing the matched base name
        grid_pattern = os.path.join(grid_folder, f"{base_name}_*_splatted2.mp4")
        grid_matches = glob.glob(grid_pattern)
        if not grid_matches:
            yield file_perc, 0, 0, f"Skipping {filename} (no grid video found)", "Running"
            continue
            
        grid_video_path = grid_matches[0] 
        match_w = re.search(rf"{base_name}_(\d+)_splatted2\.mp4", os.path.basename(grid_video_path))
        if match_w:
            w_grid = match_w.group(1)
        else:
            w_grid = "unknown"
            
        out_name = f"{base_name}_{w_grid}_inpainted_right_eye.mp4"
        out_path = os.path.join(output_folder, out_name)
        
        temp_out_path = os.path.join(output_folder, f"{base_name}_lefteye_generated.mp4")
        
        if "Option 2" in model_variant:
            model_config = "configs/m2svid_no_fullatten.yaml"
            ckpt = "ckpts/m2svid_no_full_atten_weights.pt"
        else:
            model_config = "configs/m2svid.yaml"
            ckpt = "ckpts/m2svid_weights.pt"
            
        cmd_inpaint = [
            sys.executable, "inpaint_and_refine.py",
            "--video_path", left_eye_path,
            "--grid_video_path", grid_video_path,
            "--output_folder", output_folder,
            "--mask_antialias", str(mask_antialias),
            "--spatial_tile_size", str(tile_size),
            "--spatial_tile_overlap", str(tile_overlap),
            "--chunk_size", str(chunk_size),
            "--overlap", str(overlap),
            "--original_input_blend_strength", str(original_input_blend_strength),
            "--model_config", model_config,
            "--ckpt", ckpt
        ]
        
        temp_perc = 0
        spat_perc = 0
        
        yield file_perc, temp_perc, spat_perc, f"Inpainting {filename}...", "Running"
        for sub_perc, desc in run_subprocess_with_progress(cmd_inpaint, inpaint_env, f"{base_name} - Inpainting"):
            if "Temporal" in desc:
                temp_perc = sub_perc
                if spat_perc == 100 or temp_perc == 0:
                    spat_perc = 0
            elif "Spatial" in desc:
                spat_perc = sub_perc
            
            yield file_perc, temp_perc, spat_perc, f"File {i+1}/{total_files} | {filename} - {desc}", "Running"
        
        if os.path.exists(temp_out_path):
            if os.path.exists(out_path):
                os.remove(out_path)
            os.rename(temp_out_path, out_path)
            
        # Move to finish folder
        finish_dir = os.path.join(left_eye_folder, "finish")
        os.makedirs(finish_dir, exist_ok=True)
        shutil.move(left_eye_path, os.path.join(finish_dir, filename))
        
    yield 100, 100, 100, "All files processed.", "Inpainting Section Processing Complete!"

def process_merging(
    inpainted_folder, original_folder, mask_folder, output_folder,
    use_gpu, pad_to_16_9, add_borders, resume, output_format, batch_chunk_size, enable_color_transfer,
    codec, output_crf,
    mask_binarize_threshold, mask_dilate_kernel_size, mask_blur_kernel_size,
    shadow_shift, shadow_start_opacity, shadow_opacity_decay, shadow_min_opacity, shadow_decay_gamma,
    convergence
):
    if not inpainted_folder or not os.path.exists(inpainted_folder):
        yield 0, 0, "Error: Inpainted folder invalid or does not exist.", "Failed"
        return
        
    os.makedirs(output_folder, exist_ok=True)
    
    # Build global settings (fallback for videos without per-video settings)
    global_settings = {
        "inpainted_folder": inpainted_folder,
        "original_folder": original_folder,
        "mask_folder": mask_folder,
        "output_folder": output_folder,
        "use_gpu": use_gpu,
        "pad_to_16_9": pad_to_16_9,
        "add_borders": add_borders,
        "resume": True,  # Always move to finished after processing
        "output_format": output_format,
        "batch_chunk_size": int(batch_chunk_size),
        "enable_color_transfer": enable_color_transfer,
        "codec": codec,
        "output_crf": int(output_crf),
        "mask_binarize_threshold": float(mask_binarize_threshold),
        "mask_dilate_kernel_size": int(mask_dilate_kernel_size),
        "mask_blur_kernel_size": int(mask_blur_kernel_size),
        "shadow_shift": int(shadow_shift),
        "shadow_start_opacity": float(shadow_start_opacity if shadow_start_opacity is not None else 0.7),
        "shadow_opacity_decay": float(shadow_opacity_decay if shadow_opacity_decay is not None else 0.1),
        "shadow_min_opacity": float(shadow_min_opacity if shadow_min_opacity is not None else 0.0),
        "shadow_decay_gamma": float(shadow_decay_gamma if shadow_decay_gamma is not None else 1.0),
        "convergence": int(convergence or 0),
        "encoding_quality": "Medium",
        "encoding_tune": "None",
        "color_tags": "Auto",
        "nvenc_lookahead_enabled": False,
        "nvenc_lookahead": 16,
        "nvenc_spatial_aq": False,
        "nvenc_temporal_aq": False,
        "nvenc_aq_strength": 8
    }
    
    # Scan for per-video .mergesettings.json files and merge into settings
    import glob as _glob
    inpainted_videos = sorted(_glob.glob(os.path.join(inpainted_folder, "*_inpainted_*.mp4")))
    per_video_overrides = {}
    for vid_path in inpainted_videos:
        sidecar_path = os.path.splitext(vid_path)[0] + ".mergesettings.json"
        if os.path.exists(sidecar_path):
            try:
                with open(sidecar_path, "r") as f:
                    per_video_overrides[os.path.basename(vid_path)] = json.load(f)
            except Exception:
                pass
    
    # Write the config with per-video overrides embedded
    global_settings["per_video_overrides"] = per_video_overrides
    
    config_path = os.path.join(output_folder, "temp_merge_config.json")
    with open(config_path, "w") as f:
        json.dump(global_settings, f)
        
    cmd_merge = [sys.executable, "run_merging.py", "--config", config_path]
    merge_env = os.environ.copy()
    merge_env["PYTHONUNBUFFERED"] = "1"
    
    file_perc = 0
    sub_perc = 0
    
    info_msg = f"Starting Merging — {len(per_video_overrides)} video(s) have saved settings"
    yield file_perc, sub_perc, info_msg, "Running"
    
    for perc, desc in run_subprocess_with_progress(cmd_merge, merge_env, "Merging"):
        if "Processing File" in desc:
            m = re.search(r"Processing File (\d+)/(\d+)", desc)
            if m:
                total = int(m.group(2))
                if total > 0:
                    file_perc = int((int(m.group(1)) - 1) / total * 100)
        else:
            sub_perc = perc
            
        yield file_perc, sub_perc, desc, "Running"
        
    yield 100, 100, "All files processed.", "Merging Section Processing Complete!"

with gr.Blocks(title="M2SVID Pipeline", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# M2SVID Pipeline Processing")
    
    with gr.Tabs():
        with gr.Tab("Section 1: Warping"):
            gr.Markdown("Warp videos using depth maps with optional low-res generation for inpainting.")
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        w_input_folder = gr.Textbox(label="Input Video Folder (Left Eyes: filename.mp4)", value="demo/input", scale=4)
                        w_input_btn = gr.Button("Browse", scale=1)
                    with gr.Row():
                        w_depth_folder = gr.Textbox(label="Depth Map Folder (filename_depth.mp4)", value="demo/depth", scale=4)
                        w_depth_btn = gr.Button("Browse", scale=1)
                    w_disparity = gr.Slider(minimum=0.000, maximum=0.100, value=0.035, step=0.001, label="Disparity Percentage")
                with gr.Column():
                    with gr.Row():
                        w_lefteye_folder = gr.Textbox(label="Output: Left Eye Folder (For Downscaled versions)", value="demo/lefteye", scale=4)
                        w_lefteye_btn = gr.Button("Browse", scale=1)
                    with gr.Row():
                        w_hires_folder = gr.Textbox(label="Output: High Res Warped Folder", value="demo/warped_high", scale=4)
                        w_hires_btn = gr.Button("Browse", scale=1)
                    with gr.Row():
                        w_lowres_folder = gr.Textbox(label="Output: Low Res Warped Folder", value="demo/warped_low", scale=4)
                        w_lowres_btn = gr.Button("Browse", scale=1)
                    
            with gr.Row():
                with gr.Column(variant="panel"):
                    gr.Markdown("### High Res Settings")
                    w_high_batch = gr.Number(label="High Res Batch Size", value=10, precision=0)
                    w_high_res = gr.Textbox(label="High Res Output Resolution (W x H)", value="1920x1024")

                with gr.Column(variant="panel"):
                    gr.Markdown("### Low Res Settings")
                    w_enable_low = gr.Checkbox(label="Enable Low Res Warping", value=True)
                    w_low_batch = gr.Number(label="Low Res Batch Size", value=10, precision=0)
                    w_low_res = gr.Textbox(label="Low Res Output Resolution (W x H)", value="1280x704")
                    
            with gr.Row():
                w_file_prog = gr.Slider(minimum=0, maximum=100, step=1, label="Overall File Progress (%)", interactive=False)
                w_sub_prog = gr.Slider(minimum=0, maximum=100, step=1, label="Current Stage Progress (%)", interactive=False)
                
            w_prog_text = gr.Textbox(label="Progress Details", interactive=False)
            w_btn = gr.Button("Start Batch Warping", variant="primary")
            w_output = gr.Textbox(label="Status")
            
            w_input_btn.click(fn=browse_folder, inputs=[w_input_folder], outputs=[w_input_folder])
            w_depth_btn.click(fn=browse_folder, inputs=[w_depth_folder], outputs=[w_depth_folder])
            w_lefteye_btn.click(fn=browse_folder, inputs=[w_lefteye_folder], outputs=[w_lefteye_folder])
            w_hires_btn.click(fn=browse_folder, inputs=[w_hires_folder], outputs=[w_hires_folder])
            w_lowres_btn.click(fn=browse_folder, inputs=[w_lowres_folder], outputs=[w_lowres_folder])
            
            w_btn.click(
                fn=process_warping,
                inputs=[
                    w_input_folder, w_depth_folder, w_lefteye_folder, w_hires_folder, w_lowres_folder,
                    w_disparity, w_high_batch, w_high_res, w_enable_low, w_low_batch, w_low_res
                ],
                outputs=[w_file_prog, w_sub_prog, w_prog_text, w_output]
            )

        with gr.Tab("Section 2: Inpainting and Refine"):
            gr.Markdown("Inpaint right eyes using downscaled Left Eye and Grid Video chunks.")
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        i_lefteye_folder = gr.Textbox(label="Input: Left Eye Folder (filename_lefteye.mp4)", value="demo/lefteye", scale=4)
                        i_lefteye_btn = gr.Button("Browse", scale=1)
                    with gr.Row():
                        i_grid_folder = gr.Textbox(label="Input: Grid Video Folder (e.g. filename_1280_splatted2.mp4)", value="demo/warped_low", scale=4)
                        i_grid_btn = gr.Button("Browse", scale=1)
                    with gr.Row():
                        i_output_folder = gr.Textbox(label="Output: Final Right Eye Folder", value="demo/refine_output", scale=4)
                        i_output_btn = gr.Button("Browse", scale=1)
                with gr.Column():
                    i_model_variant = gr.Radio(
                        label="Model Variant", 
                        choices=["Option 1: Full Attention", "Option 2: Without Full Attention"], 
                        value="Option 1: Full Attention"
                    )
                    i_mask_antialias = gr.Number(label="Mask Antialias", value=0, precision=0)
                    i_tile_size = gr.Number(label="Spatial Tile Size", value=256, precision=0)
                    i_tile_overlap = gr.Number(label="Spatial Tile Overlap", value=32, precision=0)
                    i_chunk_size = gr.Number(label="Chunk Size (frames per pass, max 25)", value=25, precision=0)
                    i_overlap = gr.Number(label="Overlap (Temporal Crossfade)", value=3, precision=0)
                    i_original_input_blend_strength = gr.Number(label="Original Input Blend Strength (Context)", value=0.0, step=0.1)
                    
            with gr.Row():
                i_file_prog = gr.Slider(minimum=0, maximum=100, step=1, label="Overall File Progress (%)", interactive=False)
                i_temp_prog = gr.Slider(minimum=0, maximum=100, step=1, label="Temporal Chunks Progress (%)", interactive=False)
                i_spat_prog = gr.Slider(minimum=0, maximum=100, step=1, label="Spatial Tiles Progress (%)", interactive=False)
                
            i_prog_text = gr.Textbox(label="Progress Details", interactive=False)
            i_btn = gr.Button("Start Batch Inpainting", variant="primary")
            i_output = gr.Textbox(label="Status")
            
            i_lefteye_btn.click(fn=browse_folder, inputs=[i_lefteye_folder], outputs=[i_lefteye_folder])
            i_grid_btn.click(fn=browse_folder, inputs=[i_grid_folder], outputs=[i_grid_folder])
            i_output_btn.click(fn=browse_folder, inputs=[i_output_folder], outputs=[i_output_folder])

            i_btn.click(
                fn=process_inpainting,
                inputs=[
                    i_lefteye_folder, i_grid_folder, i_output_folder,
                    i_mask_antialias, i_tile_size, i_tile_overlap, i_chunk_size, i_overlap, i_original_input_blend_strength,
                    i_model_variant
                ],
                outputs=[i_file_prog, i_temp_prog, i_spat_prog, i_prog_text, i_output]
            )

        with gr.Tab("Section 3: Merging GUI"):
            gr.Markdown("Finalize videos, merge mask outputs and encode in various SBS 3D formats.")
            
            # ---- State for scanned video list ----
            m_video_list_state = gr.State([])
            
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        m_inpainted_folder = gr.Textbox(label="Input: Inpainted Video Folder", value="demo/refine_output", scale=4)
                        m_inpainted_btn = gr.Button("Browse", scale=1)
                    with gr.Row():
                        m_original_folder = gr.Textbox(label="Input: Original Video Folder (Left Eye)", value="demo/input", scale=4)
                        m_original_btn = gr.Button("Browse", scale=1)
                    with gr.Row():
                        m_mask_folder = gr.Textbox(label="Input: Mask/Splatted Video Folder", value="demo/warped_high", scale=4)
                        m_mask_btn = gr.Button("Browse", scale=1)
                    with gr.Row():
                        m_output_folder = gr.Textbox(label="Output: Final Merged Video Folder", value="demo/merged_output", scale=4)
                        m_output_btn = gr.Button("Browse", scale=1)
                        
                with gr.Column():
                    gr.Markdown("### Processing Options")
                    m_output_format = gr.Dropdown(
                        label="Output Format",
                        choices=[
                            "Half SBS (Left-Right)", "Full SBS (Left-Right)", "Full SBS Cross-eye (Right-Left)",
                            "Double SBS", "Anaglyph (Red/Cyan)", "Anaglyph Half-Color", "Right-Eye Only"
                        ],
                        value="Half SBS (Left-Right)"
                    )
                    with gr.Row():
                        m_use_gpu = gr.Checkbox(label="Use GPU", value=True)
                        m_pad_to_16_9 = gr.Checkbox(label="Pad to 16:9", value=False)
                        m_add_borders = gr.Checkbox(label="Apply Borders", value=True)
                        m_resume = gr.Checkbox(label="Resume", value=False)
                        m_color_transfer = gr.Checkbox(label="Color Transfer", value=False)
                        
                    with gr.Row():
                        m_batch_chunk_size = gr.Number(label="GPU Frame Chunk Size", value=32, precision=0)
                        m_convergence = gr.Slider(minimum=-100, maximum=100, step=1, label="Horizontal Convergence", value=0)
                    with gr.Row():
                        m_codec = gr.Dropdown(label="FFmpeg Codec", choices=["Auto", "H.264", "H.265"], value="H.265")
                        m_output_crf = gr.Number(label="Output CRF (Quality)", value=23, precision=0)
            
            with gr.Row():
                with gr.Column(variant="panel"):
                    gr.Markdown("### Mask Processing & Thresholding")
                    m_mask_bin_thresh = gr.Slider(minimum=-1.0, maximum=1.0, step=0.01, label="Mask Binarize Threshold (-1 = disabled)", value=-1.0)
                    m_mask_dilate = gr.Slider(minimum=0, maximum=50, step=1, label="Mask Dilate Kernel", value=0)
                    m_mask_blur = gr.Slider(minimum=0, maximum=50, step=1, label="Mask Blur Kernel", value=0)
                with gr.Column(variant="panel"):
                    gr.Markdown("### Shadow / Edge Mitigation")
                    m_shadow_shift = gr.Slider(minimum=0, maximum=100, step=1, label="Shadow Shift Amount", value=0)
                    m_shadow_start_op = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="Shadow Start Opacity", value=0.7)
                    m_shadow_decay = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="Shadow Opacity Decay", value=0.1)
                    m_shadow_min_op = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="Shadow Min Opacity", value=0.0)
                    m_shadow_gamma = gr.Slider(minimum=0.1, maximum=5.0, step=0.1, label="Shadow Decay Gamma", value=1.0)
            
            gr.Markdown("---")
            gr.Markdown("### Video Preview & Selection")
            
            with gr.Row():
                m_scan_btn = gr.Button("🔍 Scan Videos", variant="secondary")
                m_video_dropdown = gr.Dropdown(label="Select Video", choices=[], interactive=True, scale=3)
                m_video_info = gr.Textbox(label="Video Info", interactive=False, scale=2)
            
            with gr.Row():
                with gr.Column(scale=3):
                    m_preview_image = gr.Image(label="Preview", type="pil", height=480)
                with gr.Column(scale=1):
                    m_preview_source = gr.Dropdown(
                        label="Preview Source",
                        choices=[
                            "Blended Right Eye", "Original Left Eye", "Warped Right BG",
                            "Inpainted Right Eye", "Processed Mask", "Full SBS", "Anaglyph"
                        ],
                        value="Blended Right Eye"
                    )
                    m_frame_slider = gr.Slider(minimum=0, maximum=100, step=1, label="Frame #", value=0)
                    m_preview_btn = gr.Button("🖼️ Update Preview", variant="secondary")
                    gr.Markdown("---")
                    m_save_settings_btn = gr.Button("💾 Save Settings for This Video")
                    m_load_settings_btn = gr.Button("📂 Load Settings for This Video")
                    m_settings_status = gr.Textbox(label="Settings Status", interactive=False)

            gr.Markdown("---")
            with gr.Row():
                m_file_prog = gr.Slider(minimum=0, maximum=100, step=1, label="Overall File Progress (%)", interactive=False)
                m_sub_prog = gr.Slider(minimum=0, maximum=100, step=1, label="Video Rendering Progress (%)", interactive=False)
                
            m_prog_text = gr.Textbox(label="Progress Details", interactive=False)
            m_btn = gr.Button("🚀 Start Batch Merging", variant="primary")
            m_output = gr.Textbox(label="Status")
            
            # ---- Event handlers ----
            m_inpainted_btn.click(fn=browse_folder, inputs=[m_inpainted_folder], outputs=[m_inpainted_folder])
            m_original_btn.click(fn=browse_folder, inputs=[m_original_folder], outputs=[m_original_folder])
            m_mask_btn.click(fn=browse_folder, inputs=[m_mask_folder], outputs=[m_mask_folder])
            m_output_btn.click(fn=browse_folder, inputs=[m_output_folder], outputs=[m_output_folder])
            
            # Scan videos
            def do_scan_videos(inpainted_f, mask_f, original_f):
                from merge_preview import scan_videos
                vlist = scan_videos(inpainted_f, mask_f, original_f)
                if not vlist:
                    return [], gr.update(choices=[], value=None), "No valid videos found. Make sure inpainted folder contains *_inpainted_right_eye.mp4 files."
                names = [v["base_name"] for v in vlist]
                return vlist, gr.update(choices=names, value=names[0]), f"Found {len(vlist)} video(s)"
            
            m_scan_btn.click(
                fn=do_scan_videos,
                inputs=[m_inpainted_folder, m_mask_folder, m_original_folder],
                outputs=[m_video_list_state, m_video_dropdown, m_video_info]
            )
            
            # Update preview
            def do_preview(video_list, selected_video, frame_idx, preview_source,
                           inpainted_folder, original_folder, mask_folder,
                           use_gpu, add_borders, color_transfer,
                           mask_thresh, mask_dilate, mask_blur,
                           shadow_shift, shadow_start_op, shadow_decay, shadow_min_op, shadow_gamma, convergence):
                if not video_list or not selected_video:
                    return None, 0
                
                video_info = None
                for v in video_list:
                    if v["base_name"] == selected_video:
                        video_info = v
                        break
                if not video_info:
                    return None, 0
                
                from merge_preview import generate_preview_frame
                settings = {
                    "inpainted_folder": inpainted_folder,
                    "original_folder": original_folder,
                    "mask_folder": mask_folder,
                    "use_gpu": use_gpu,
                    "add_borders": add_borders,
                    "enable_color_transfer": color_transfer,
                    "mask_binarize_threshold": float(mask_thresh if mask_thresh is not None else -1.0),
                    "mask_dilate_kernel_size": int(mask_dilate or 0),
                    "mask_blur_kernel_size": int(mask_blur or 0),
                    "shadow_shift": int(shadow_shift or 0),
                    "shadow_start_opacity": float(shadow_start_op if shadow_start_op is not None else 0.7),
                    "shadow_opacity_decay": float(shadow_decay if shadow_decay is not None else 0.1),
                    "shadow_min_opacity": float(shadow_min_op if shadow_min_op is not None else 0.0),
                    "shadow_decay_gamma": float(shadow_gamma if shadow_gamma is not None else 1.0),
                    "convergence": int(convergence or 0),
                    "preview_source": preview_source,
                }
                try:
                    img, total_frames = generate_preview_frame(video_info, settings, int(frame_idx))
                    return img, gr.update(maximum=max(total_frames - 1, 0))
                except Exception as e:
                    import traceback
                    print(f"Preview error: {e}")
                    traceback.print_exc()
                    return None, 0
            
            m_preview_btn.click(
                fn=do_preview,
                inputs=[
                    m_video_list_state, m_video_dropdown, m_frame_slider, m_preview_source,
                    m_inpainted_folder, m_original_folder, m_mask_folder,
                    m_use_gpu, m_add_borders, m_color_transfer,
                    m_mask_bin_thresh, m_mask_dilate, m_mask_blur,
                    m_shadow_shift, m_shadow_start_op, m_shadow_decay, m_shadow_min_op, m_shadow_gamma,
                    m_convergence
                ],
                outputs=[m_preview_image, m_frame_slider]
            )
            
            # Auto-preview on video selection change
            def on_video_change(video_list, selected_video, preview_source,
                                inpainted_folder, original_folder, mask_folder,
                                use_gpu, add_borders, color_transfer,
                                mask_thresh, mask_dilate, mask_blur,
                                shadow_shift, shadow_start_op, shadow_decay, shadow_min_op, shadow_gamma, convergence):
                return do_preview(video_list, selected_video, 0, preview_source,
                                  inpainted_folder, original_folder, mask_folder,
                                  use_gpu, add_borders, color_transfer,
                                  mask_thresh, mask_dilate, mask_blur,
                                  shadow_shift, shadow_start_op, shadow_decay, shadow_min_op, shadow_gamma, convergence)
            
            m_video_dropdown.change(
                fn=on_video_change,
                inputs=[
                    m_video_list_state, m_video_dropdown, m_preview_source,
                    m_inpainted_folder, m_original_folder, m_mask_folder,
                    m_use_gpu, m_add_borders, m_color_transfer,
                    m_mask_bin_thresh, m_mask_dilate, m_mask_blur,
                    m_shadow_shift, m_shadow_start_op, m_shadow_decay, m_shadow_min_op, m_shadow_gamma,
                    m_convergence
                ],
                outputs=[m_preview_image, m_frame_slider]
            )
            
            # Auto-preview on any parameter change
            _auto_preview_inputs = [
                m_video_list_state, m_video_dropdown, m_frame_slider, m_preview_source,
                m_inpainted_folder, m_original_folder, m_mask_folder,
                m_use_gpu, m_add_borders, m_color_transfer,
                m_mask_bin_thresh, m_mask_dilate, m_mask_blur,
                m_shadow_shift, m_shadow_start_op, m_shadow_decay, m_shadow_min_op, m_shadow_gamma,
                m_convergence
            ]
            _auto_preview_outputs = [m_preview_image, m_frame_slider]
            
            for _ctrl in [m_preview_source, m_mask_bin_thresh, m_mask_dilate, m_mask_blur,
                          m_shadow_shift, m_shadow_start_op, m_shadow_decay, m_shadow_min_op, m_shadow_gamma,
                          m_use_gpu, m_add_borders, m_color_transfer, m_convergence, m_frame_slider]:
                _ctrl.change(
                    fn=do_preview,
                    inputs=_auto_preview_inputs,
                    outputs=_auto_preview_outputs
                )
            
            # Save per-video settings
            def save_video_settings(video_list, selected_video,
                                    mask_thresh, mask_dilate, mask_blur,
                                    shadow_shift, shadow_start_op, shadow_decay, shadow_min_op, shadow_gamma,
                                    convergence, output_format, use_gpu, color_transfer, add_borders):
                if not video_list or not selected_video:
                    return "No video selected."
                
                video_info = None
                for v in video_list:
                    if v["base_name"] == selected_video:
                        video_info = v
                        break
                if not video_info:
                    return "Video not found."
                
                settings_to_save = {
                    "mask_binarize_threshold": float(mask_thresh if mask_thresh is not None else -1.0),
                    "mask_dilate_kernel_size": int(mask_dilate or 0),
                    "mask_blur_kernel_size": int(mask_blur or 0),
                    "shadow_shift": int(shadow_shift or 0),
                    "shadow_start_opacity": float(shadow_start_op if shadow_start_op is not None else 0.7),
                    "shadow_opacity_decay": float(shadow_decay if shadow_decay is not None else 0.1),
                    "shadow_min_opacity": float(shadow_min_op if shadow_min_op is not None else 0.0),
                    "shadow_decay_gamma": float(shadow_gamma if shadow_gamma is not None else 1.0),
                    "convergence": int(convergence or 0),
                    "output_format": output_format,
                    "use_gpu": use_gpu,
                    "enable_color_transfer": color_transfer,
                    "add_borders": add_borders,
                }
                
                sidecar_path = os.path.splitext(video_info["inpainted"])[0] + ".mergesettings.json"
                with open(sidecar_path, "w") as f:
                    json.dump(settings_to_save, f, indent=2)
                return f"✅ Saved settings to {os.path.basename(sidecar_path)}"
            
            m_save_settings_btn.click(
                fn=save_video_settings,
                inputs=[
                    m_video_list_state, m_video_dropdown,
                    m_mask_bin_thresh, m_mask_dilate, m_mask_blur,
                    m_shadow_shift, m_shadow_start_op, m_shadow_decay, m_shadow_min_op, m_shadow_gamma,
                    m_convergence, m_output_format, m_use_gpu, m_color_transfer, m_add_borders
                ],
                outputs=[m_settings_status]
            )
            
            # Load per-video settings
            def load_video_settings(video_list, selected_video):
                if not video_list or not selected_video:
                    return [-1.0, 0, 0, 0, 0.7, 0.1, 0.0, 1.0, 0, "Half SBS (Left-Right)", True, False, True, "No video selected."]
                
                video_info = None
                for v in video_list:
                    if v["base_name"] == selected_video:
                        video_info = v
                        break
                if not video_info:
                    return [-1.0, 0, 0, 0, 0.7, 0.1, 0.0, 1.0, 0, "Half SBS (Left-Right)", True, False, True, "Video not found."]
                
                sidecar_path = os.path.splitext(video_info["inpainted"])[0] + ".mergesettings.json"
                if not os.path.exists(sidecar_path):
                    return [-1.0, 0, 0, 0, 0.7, 0.1, 0.0, 1.0, 0, "Half SBS (Left-Right)", True, False, True,
                            f"No saved settings found for {selected_video}"]
                
                try:
                    with open(sidecar_path, "r") as f:
                        s = json.load(f)
                    return [
                        s.get("mask_binarize_threshold", -1.0),
                        s.get("mask_dilate_kernel_size", 0),
                        s.get("mask_blur_kernel_size", 0),
                        s.get("shadow_shift", 0),
                        s.get("shadow_start_opacity", 0.7),
                        s.get("shadow_opacity_decay", 0.1),
                        s.get("shadow_min_opacity", 0.0),
                        s.get("shadow_decay_gamma", 1.0),
                        s.get("convergence", 0),
                        s.get("output_format", "Half SBS (Left-Right)"),
                        s.get("use_gpu", True),
                        s.get("enable_color_transfer", False),
                        s.get("add_borders", True),
                        f"✅ Loaded settings from {os.path.basename(sidecar_path)}"
                    ]
                except Exception as e:
                    return [-1.0, 0, 0, 0, 0.7, 0.1, 0.0, 1.0, 0, "Half SBS (Left-Right)", True, False, True,
                            f"Error loading settings: {e}"]
            
            m_load_settings_btn.click(
                fn=load_video_settings,
                inputs=[m_video_list_state, m_video_dropdown],
                outputs=[
                    m_mask_bin_thresh, m_mask_dilate, m_mask_blur,
                    m_shadow_shift, m_shadow_start_op, m_shadow_decay, m_shadow_min_op, m_shadow_gamma,
                    m_convergence, m_output_format, m_use_gpu, m_color_transfer, m_add_borders,
                    m_settings_status
                ]
            )
            
            # Batch merging
            m_btn.click(
                fn=process_merging,
                inputs=[
                    m_inpainted_folder, m_original_folder, m_mask_folder, m_output_folder,
                    m_use_gpu, m_pad_to_16_9, m_add_borders, m_resume, m_output_format, m_batch_chunk_size, m_color_transfer,
                    m_codec, m_output_crf,
                    m_mask_bin_thresh, m_mask_dilate, m_mask_blur,
                    m_shadow_shift, m_shadow_start_op, m_shadow_decay, m_shadow_min_op, m_shadow_gamma,
                    m_convergence
                ],
                outputs=[m_file_prog, m_sub_prog, m_prog_text, m_output]
            )



if __name__ == "__main__":
    demo.queue().launch(inbrowser=True)
