# 本轮GPU推理审核

## astra

verified_configuration: 34frames832x480/12fps,30steps,forward+camera_r;full decode and coherent short room/camera motion,visible contrast softening after reference frame

## easyanimate

verified_configuration: V5.1-7B-InP,30steps34frames768x512/16fps;full decode and coherent forward room approach;minor generated-detail changes

## lingbot-video-dense-121f

quality_concern: This separate official-prompt121frame40step,batch_cfg configuration has clean visually reviewed output at832x480/24fps. Family remains quality_concern because prior49frame30step forest artifacts are unresolved;new success does not overwrite failure.

## hyperflow-t2va-smoke

verified_inference_only: t2va,8step model,124frames640x384/24fps;full video/audio decode,visually coherent room/forward motion. Audio32kHz stereo5.175s,finite/non-silent/no clipping;subjective audio quality and A/V semantic alignment still pending.

## hyperflow-fl2va-smoke

verified_inference_only: fl2va,8step model,124frames640x384/24fps;full video/audio decode,visually coherent room/forward motion. Audio32kHz stereo5.175s,finite/non-silent/no clipping;subjective audio quality and A/V semantic alignment still pending.

## hyperflow-ref2va-smoke

verified_inference_only: ref2va,8step model,124frames640x384/24fps;full video/audio decode,visually coherent room/forward motion. Audio32kHz stereo5.175s,finite/non-silent/no clipping;subjective audio quality and A/V semantic alignment still pending.

## allegro-native

Allegro-TI2V native88frames1280x720/15fps,100steps;full decode and reviewed coherent room with forward approach and modest turn;new details generated,reference preserved at start.

## allegro-ti2v: allegro-native

verified_configuration: Allegro-TI2V native88frames1280x720/15fps,100steps;full decode and reviewed coherent room with forward approach and modest turn;new details generated,reference preserved at start. Same concrete checkpoint/run as allegro;shared evidence for alias/family label,not another GPU run or all-family validation.

## easyanimate-i2v: easyanimate

verified_configuration: V5.1-7B-InP,30steps34frames768x512/16fps;full decode and coherent forward room approach;minor generated-detail changes Same concrete checkpoint/run as easyanimate;shared evidence for alias/family label,not another GPU run or all-family validation.
