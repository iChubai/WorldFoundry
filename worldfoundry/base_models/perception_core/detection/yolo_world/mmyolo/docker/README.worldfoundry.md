# Not a supported WorldFoundry build path

The Dockerfiles in this vendored tree are upstream MMYOLO artifacts preserved
as part of the YOLO-World snapshot described in `../../UPSTREAM.md`:

- `docker/Dockerfile`
- `docker/Dockerfile_deployment`
- `../.circleci/docker/Dockerfile`

WorldFoundry does not build or publish these images. They retain legacy
upstream bases and package assumptions, including PyTorch 1.8/1.9-era CUDA
images and Ubuntu 18.04 apt-key setup, and they are not maintained against the
current WorldFoundry runtime.

Use the repository-root `docker/Dockerfile` through
`docker/build_with_docker.sh` for the supported WorldFoundry container path.
Keep the vendored Dockerfiles unchanged so the snapshot remains comparable
with upstream. If an MMYOLO-specific image is required, build it from the
current upstream project rather than patching this vendored copy.
