import gradio as gr
import torchvision.transforms as TT
from panorama import TwinDiffusion, seed_everything

seed_everything(-1)
td = TwinDiffusion('cuda', '2.0')


def generate_panorama_fn(prompt, width, lam, view_stride, cross_time):
    image = td.text2panorama_optm(
        prompts=prompt,
        negative_prompts="",
        width=width,
        lam=lam,
        view_stride=view_stride,
        cross_time=cross_time
    )
    image = TT.ToPILImage()(image[0])
    return image


description = """
<h1 style="text-align: center;">Generate Panoramic Images with TwinDiffusion</h1>
<h3 style="text-align: center;"><a href="https://github.com/0606zt/TwinDiffusion" target="blank"> [Code]</a> 
<a href="https://arxiv.org/abs/2404.19475" target="blank"> [Paper]</a></h3>
"""

prompt_exp = [
    ["A photo of the dolomites"],
    ["A photo of the mountain, lake, people and boats"],
    ["A landscape ink painting"],
    ["Natural landscape in anime style illustration"],
    ["A graphite sketch of a majestic mountain range"],
    ["A surrealistic artwork of urban park at dawn"]
]

with gr.Blocks() as demo:
    gr.Markdown(description)
    output = gr.Image(label="Generated Image")
    with gr.Row():
        with gr.Column(scale=3):
            prompt = gr.Textbox(label="Enter your prompt")
        with gr.Column(scale=1):
            generate_btn = gr.Button("Generate")
    gr.Examples(examples=prompt_exp, inputs=prompt)

    with gr.Accordion(label="⚙️ More Settings", open=False):
        width = gr.Slider(
            label="Image Width",
            minimum=512,
            maximum=4608,
            step=128,
            value=2048)
        lam = gr.Slider(
            label="Lambda",
            minimum=0,
            maximum=100,
            step=1,
            value=1)
        view_stride = gr.Slider(
            label="View Stride",
            minimum=8,
            maximum=48,
            step=8,
            value=16)
        cross_time = gr.Slider(
            label="Cross Time",
            minimum=2,
            maximum=10,
            step=1,
            value=2)

    generate_btn.click(
        fn=generate_panorama_fn,
        inputs=[prompt, width, lam, view_stride, cross_time],
        outputs=output
    )
demo.launch(share=True)
