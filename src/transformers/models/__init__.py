# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from . import (
    albert,
    align,
    altclip,
    aria,
    audio_spectrogram_transformer,
    auto,
    autoformer,
    bamba,
    bark,
    bart,
    barthez,
    bartpho,
    beit,
    bert,
    bert_generation,
    bert_japanese,
    bertweet,
    big_bird,
    bigbird_pegasus,
    biogpt,
    bit,
    blenderbot,
    blenderbot_small,
    blip,
    blip_2,
    bloom,
    bridgetower,
    bros,
    byt5,
    camembert,
    canine,
    chameleon,
    chinese_clip,
    clap,
    clip,
    clipseg,
    clvp,
    code_llama,
    codegen,
    cohere,
    cohere2,
    colpali,
    conditional_detr,
    convbert,
    convnext,
    convnextv2,
    cpm,
    cpmant,
    ctrl,
    cvt,
    dab_detr,
    dac,
    data2vec,
    dbrx,
    deberta,
    deberta_v2,
    decision_transformer,
    deformable_detr,
    deit,
    deprecated,
    depth_anything,
    detr,
    dialogpt,
    dinat,
    dinov2,
    dinov2_with_registers,
    distilbert,
    dit,
    donut,
    dpr,
    dpt,
    efficientnet,
    electra,
    encodec,
    encoder_decoder,
    ernie,
    esm,
    falcon,
    falcon_mamba,
    fastspeech2_conformer,
    flaubert,
    flava,
    fnet,
    focalnet,
    fsmt,
    funnel,
    fuyu,
    gemma,
    gemma2,
    git,
    glm,
    glpn,
    gpt2,
    gpt_bigcode,
    gpt_neo,
    gpt_neox,
    gpt_neox_japanese,
    gpt_sw3,
    gptj,
    granite,
    granitemoe,
    grounding_dino,
    groupvit,
    herbert,
    hiera,
    hubert,
    ibert,
    idefics,
    idefics2,
    idefics3,
    ijepa,
    imagegpt,
    informer,
    instructblip,
    instructblipvideo,
    jamba,
    jetmoe,
    kosmos2,
    layoutlm,
    layoutlmv2,
    layoutlmv3,
    layoutxlm,
    led,
    levit,
    lilt,
    llama,
    llava,
    llava_next,
    llava_next_video,
    llava_onevision,
    longformer,
    longt5,
    luke,
    lxmert,
    m2m_100,
    mamba,
    mamba2,
    marian,
    markuplm,
    mask2former,
    maskformer,
    mbart,
    mbart50,
    megatron_bert,
    megatron_gpt2,
    mgp_str,
    mimi,
    mistral,
    mixtral,
    mllama,
    mluke,
    mobilebert,
    mobilenet_v1,
    mobilenet_v2,
    mobilevit,
    mobilevitv2,
    modernbert,
    moshi,
    mpnet,
    mpt,
    mra,
    mt5,
    musicgen,
    musicgen_melody,
    mvp,
    myt5,
    nemotron,
    nllb,
    nllb_moe,
    nougat,
    nystromformer,
    olmo,
    olmo2,
    olmoe,
    omdet_turbo,
    oneformer,
    openai,
    opt,
    owlv2,
    owlvit,
    paligemma,
    patchtsmixer,
    patchtst,
    pegasus,
    pegasus_x,
    perceiver,
    persimmon,
    phi,
    phi3,
    phimoe,
    phobert,
    pix2struct,
    pixtral,
    plbart,
    poolformer,
    pop2piano,
    prophetnet,
    pvt,
    pvt_v2,
    qwen2,
    qwen2_audio,
    qwen2_moe,
    qwen2_vl,
    rag,
    recurrent_gemma,
    reformer,
    regnet,
    rembert,
    resnet,
    roberta,
    roberta_prelayernorm,
    roc_bert,
    roformer,
    rt_detr,
    rwkv,
    sam,
    seamless_m4t,
    seamless_m4t_v2,
    segformer,
    seggpt,
    sew,
    sew_d,
    siglip,
    speech_encoder_decoder,
    speech_to_text,
    speecht5,
    splinter,
    squeezebert,
    stablelm,
    starcoder2,
    superpoint,
    swiftformer,
    swin,
    swin2sr,
    swinv2,
    switch_transformers,
    t5,
    table_transformer,
    tapas,
    time_series_transformer,
    timesformer,
    timm_backbone,
    timm_wrapper,
    trocr,
    tvp,
    udop,
    umt5,
    unispeech,
    unispeech_sat,
    univnet,
    upernet,
    video_llava,
    videomae,
    vilt,
    vipllava,
    vision_encoder_decoder,
    vision_text_dual_encoder,
    visual_bert,
    vit,
    vit_mae,
    vit_msn,
    vitdet,
    vitmatte,
    vits,
    vivit,
    wav2vec2,
    wav2vec2_bert,
    wav2vec2_conformer,
    wav2vec2_phoneme,
    wav2vec2_with_lm,
    wavlm,
    whisper,
    x_clip,
    xglm,
    xlm,
    xlm_roberta,
    xlm_roberta_xl,
    xlnet,
    xmod,
    yolos,
    yoso,
    zamba,
    zoedepth,
)
