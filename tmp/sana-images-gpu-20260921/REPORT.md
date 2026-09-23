
## sana-600m-512px

verified_configuration: Native512,20steps,bf16;raw sample/latents finite,strict model load,red teapot/white cup and sunlit wooden-table semantics verified;handle has a minor generated shape irregularity.

## sana-600m-1024px

verified_configuration: Native1024,20steps,bf16;raw sample/latents finite,strict model load,red ceramic teapot and white cup by sunlit window verified.

## sana1p5-4800m-1024px

Distinct4.8B,20steps,native1024,bf16;strict load,finite sample/latents,visually correct red teapot/white cup on sunlit wooden table.

## sana-1600m-512px: sana-1600m-512px

quality_concern: 512px standard checkpoint:finite clear image but duplicated teapot spout. Visually reviewed;20step native Euler,single seed.

## sana-1600m-1024px-multiling: sana-1600m-1024px-multiling

quality_concern: 1024px multilingual checkpoint:finite image but extra pots and duplicated spouts;English prompt only. Visually reviewed;20step native Euler,single seed.

## sana-sprint-1600m-1024px

quality_concern: Two-step1024x1024,BF16 after mask fix and verified teacher download;finite raw output and strict loads. Actual preview contains two red pots plus cup instead of one teapot and one cup;left vessel has two handle-like appendages. Single prompt/seed quality concern.
