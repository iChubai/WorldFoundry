# EgoWM 3-DoF repair and 25-DoF regression

The WorldFoundry launcher now selects the official UNet/pipeline by checkpoint structure. The 3-DoF checkpoint has a 96-wide action embedding and no state embedding; the 25-DoF checkpoint has an 800-wide action embedding plus state embedding. Both load strictly and produce nine decodable 512×512 frames on separate H100 GPUs.

The official CMU-click hallway image and a synthetic +0.05 first-component action were used for each eight-frame rollout. The 25-DoF branch preserves the hallway with modest viewpoint change. The 3-DoF four-step result collapses into blur. Repeating 3-DoF at the upstream default 25 sampling steps avoids that collapse but changes room geometry abruptly and adds a person. This image is outside the checkpoint's stated RECON/SCAND/Tartan in-domain tests, and no navigation accuracy or task success can be inferred.

Evidence: `egowm-3dof-fixed/status.json`, `egowm-3dof-fixed/generated-25steps.mp4`, `egowm-3dof-fixed/generated-25steps.log`, `egowm-3dof-fixed/quality-review.json`, `egowm-25dof-regression/status.json`. The earlier 3-DoF shape-mismatch log is in `egowm-3dof/generated.mp4.log`.
