FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04 AS base
#from nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as base:
ADD . /llmseg
WORKDIR /llmseg
RUN chmod +x install_deps.sh & install_deps.sh
RUN HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download JCdesu/LLM-Seg-deepspeed

WORKDIR models
#llama weights downloader
RUN git clone https://github.com/shawwn/llama-dl.git
WORKDIR llama-dl
# select the 7B model for download
RUN git apply ../llama_dl_patch.patch
RUN ./llama.sh
RUN mv 7B ..
RUN wget https://huggingface.co/liuhaotian/LLaVA-Lightning-7B-delta-v1-1

# Apply delta weights
RUN wget  https://huggingface.co/liuhaotian/LLaVA-Lightning-7B-delta-v1-1
RUN export PYTHONPATH="/llmseg/model/llava/$PYTHONPATH"
RUN python3 -m llava.model.apply_delta --base /llmseg/llama-dl/7B_transformers/ --target LLaVA-Lightning-7B --delta liuhaotian/LLaVA-Lightning-7B-delta-v1-1

#Keep the container alive
CMD ["tail", "-f", "/dev/null"]
