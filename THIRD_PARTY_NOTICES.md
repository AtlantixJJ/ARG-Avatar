# Third-party notices

The root [CC BY-NC-SA 4.0 license](LICENSE) covers original ARGAvatar source code,
the ARGAvatar model weights, and our modifications to them. It does not replace the licenses of the components below.
Existing copyright and license headers in individual files remain applicable.

| Component | Included license / upstream terms |
| --- | --- |
| `xlib/model/sam_3d_body/` | [SAM License](xlib/model/sam_3d_body/LICENSE), copied from facebookresearch/sam-3d-body commit `b5c765a0d89d789985e186d396315e7590887b94`. ARGAvatar modifies its model wrappers and runtime integration. |
| `thirdparty/MHR/`, `mesh_exports/mhr_mesh_lod1.fbx` | [MHR Apache-2.0 license](thirdparty/MHR/LICENSE). The FBX is the MHR mesh layout used by the model. |
| `thirdparty/sam2/` | [SAM 2 license](thirdparty/sam2/LICENSE); additional notices are preserved in that directory. |
| `thirdparty/simple-knn/` | [License](thirdparty/simple-knn/LICENSE.md), including its non-commercial terms |
| `thirdparty/diff-gaussian-rasterization-da/` | [License](thirdparty/diff-gaussian-rasterization-da/LICENSE.md), including its non-commercial terms; [GLM notices](thirdparty/diff-gaussian-rasterization-da/third_party/glm/copying.txt) |

Downloaded components keep their upstream terms: [SAM 3D Body](https://huggingface.co/facebook/sam-3d-body-dinov3/blob/main/LICENSE),
[DINOv3](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md),
[nvdiffrast](https://github.com/NVlabs/nvdiffrast/blob/main/LICENSE),
[detectron2](https://github.com/facebookresearch/detectron2/blob/main/LICENSE),
[MoGe](https://github.com/microsoft/MoGe/blob/main/LICENSE), and optional
[Kaolin](https://github.com/NVIDIAGameWorks/kaolin/blob/master/LICENSE).
The pinned DINOv3 source download includes its license alongside its code.

The curated dataset images under `assets/example_images/` retain their dataset
terms, and so do the renders of dataset subjects in `assets/gallery/` (the README
gallery); see the [LICENSE](LICENSE) header. They are not licensed under
CC BY-NC-SA 4.0. The full NeuMan imagery is downloaded from Apple
by the user; it is absent from the bundled detection archive. That archive contains
only this project's masks, pose/camera records, and validation hierarchy, and is
accompanied by its [manifest](assets/benchmark/argavatar-neuman-detections.tar.gz.sha256).
