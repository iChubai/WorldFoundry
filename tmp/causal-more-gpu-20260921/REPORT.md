# 本轮真实GPU推理审核

## self-forcing-dmd

verified_configuration: DMD checkpoint,21latent->81RGB frames832x480/16fps;full decode,coherent workshop/person and forward camera review

证据：tmp/causal-more-gpu-20260921/self-forcing-dmd/status.json；tmp/causal-more-gpu-20260921/self-forcing-dmd/demo.validation.json。接触图已人工查看。

## lingbot-video-dense

quality_concern: Dense1.3B,30steps49frames832x480/24fps;full decode passes,but conspicuous stripe/checker artifacts,oversaturation and unstable detail;VAE verified float32 already;cause unlocalized

证据：tmp/causal-more-gpu-20260921/lingbot-video-dense/status.json；tmp/causal-more-gpu-20260921/lingbot-video-dense/demo.validation.json。接触图已人工查看。

## worldcam

verified_configuration: 8ARsteps,65conditioning frames trimmed,32newframes832x480/30fps,official game video+palindrome intrinsics/extrinsics,50steps;full decode and coherent camera/game view review;short continuation only

## lingbot-world-v2-compact

verified_configuration: 14B causal-fast,49frames848x464/16fps,forward action;full decode and coherent room/subtle forward progression review;no long-memory claim

## zing-i2v

verified_configuration: Zing0.5,81frames832x480/24fps,81explicit forward-key rows;exact WanModel/TextEncoder/VAE loads,full decode,coherent scene with modest forward movement

## alaya-v11-ar

verified_configuration: AR stage2b,1round33frames960x544/24fps,default C2W forward0.0049/frame;full decode and coherent short-scene motion review;external source hashes captured

## alaya-v11-dmd-multiround

verified_configuration: DMD stage3,2rounds65frames960x544/24fps,default C2W forward0.0049/frame;full decode,coherent seam/short rollout visual review;does not establish long memory robustness;external source hashes captured

## echo-base

verified_configuration: Base distinct weights,49frames1280x704/24fps,w-48 action;full decode,coherent room and forward approach;minor generated details change

## echo-flash

verified_configuration: Flash distinct weights,49frames1280x704/24fps,w-48 action;full decode,coherent room/forward approach,limited minor detail changes

## minwm-wan

quality_concern: 77frames832x480/16fps,w*19;full decode but explicit first-person prompt becomes third-person with full character visible;no longer classify semantic pass

## zing-strafe

verified_configuration: 81frames832x480/24fps,explicit strafe controls;full decode,clear lateral parallax distinct from same-seed forward case,scene remains coherent

## astronex-bidirectional

verified_configuration: Bidirectional,17latent->65RGB832x480/24fps,w*9,h*8;full decode,forward progression then turning while scene preserved

## alaya-native

verified_configuration: Native original checkpoint,2rounds64frames960x544/24fps,w*64;full decode and coherent fantasy room/camera motion;seam visually continuous,minor lighting drift;short rollout only.

## magicworld-fast

verified_configuration: Fast model33frames832x480/16fps,explicit33native pose rows translating forward;full decode and coherent room approach with modest object detail drift.

## astronex-causal

Causal4step,93frames832x480/24fps,w*12,h*12;full decode and reviewed forward approach then turn,coherent short room scene

## SolarWM237f ema

verified_configuration: 237frames864x480/16fps,EMA4step full14.8s rollout;full decode and normal/jump-frame visual review shows coherent forward approach to wall,minor scene deformation.

## SolarWM237f live

quality_concern: 237frames864x480/16fps,live4step full14.8s rollout;full decode finite but frames227-233 degenerate into rapidly flashing pink/green foreground blobs;max adjacent frame delta58.39.EMA comparison stays coherent;long-horizon live branch is not a quality pass.
