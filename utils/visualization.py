# When using WSL do the following commands: 
# export DISPLAY=:0
# export XDG_SESSION_TYPE=x11

# Can be used for fast visualization but you can follow
# the script in helimap dataset preprocessing code and 
# the viz3d.py to have better tools 

import numpy as np
try:
    import open3d as o3d
except ImportError:
    o3d = None

def vizualize_point_cloud_with_colors(points, rgb):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    # Create coordinate frame at origin
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=2.0,   # adjust size depending on your scene scale
        origin=[0, 0, 0]
    )

    o3d.visualization.draw_geometries([pcd, frame])

# if __name__ == "__main__":
#     def _voxel_downsample(pts, rgb, labels, voxel_size):
#         """Keep one point per voxel (first-point rule, O(N log N)).
#         pts, rgb, labels are in the same row order. Returns downsampled arrays."""
#         vox = np.floor(pts / voxel_size).astype(np.int64)
#         vox -= vox.min(axis=0)
#         mx = vox.max(axis=0) + 1
#         keys = vox[:, 0] * (mx[1] * mx[2]) + vox[:, 1] * mx[2] + vox[:, 2]
#         _, keep = np.unique(keys, return_index=True)
#         keep.sort()
#         return pts[keep], rgb[keep], (labels[keep] if labels is not None else None)

#     def _load_zone_points(las_path):
#         """
#         Returns:
#             xyz    : (N, 3) float64  absolute world coords (X*scale + offset)
#             rgb    : (N, 3) float32  LAS RGB in [0, 255]
#             labels : (N,)   int32    raw class IDs, or None if absent (test set)
#         """
#         import laspy
#         las = laspy.read(las_path)
#         xyz = np.vstack((las.X, las.Y, las.Z)).T * las.header.scale + las.header.offset
#         rgb = np.stack([las.red, las.green, las.blue], axis=1).astype(np.float32) / 65535 * 255
#         if 'ground_truth' in las.point_format.extra_dimension_names:
#             labels = np.array(las['ground_truth'], dtype=np.int32)
#         else:
#             labels = None
#         return xyz, rgb, labels

#     def _load_zone_points_helimap(las_path):
#         """
#         Returns:
#             xyz    : (N, 3) float64  absolute world coords (X*scale + offset)
#             rgb    : (N, 3) float32  LAS RGB in [0, 255]
#             labels : (N,)   int32    raw class IDs, or None if absent (test set)
#         """
#         import laspy
#         las = laspy.read(las_path)
#         xyz = np.vstack((las.X, las.Y, las.Z)).T * las.header.scale + las.header.offset
#         rgb = np.stack([las.red, las.green, las.blue], axis=1).astype(np.float32) / 65535 * 255
#         labels = np.array(las['classification'], dtype=np.int32)
#         return xyz, rgb, labels
    # path = "chunk_path"
    # path_xyz = path+"/xyz.npy"
    # path_rgb = path+"/rgb.npy"
    # xyz = np.load(path_xyz)
    # rgb = np.load(path_rgb)/255

    # print(xyz.shape)
    # vizualize_point_cloud_with_colors(xyz, rgb)
    # Load data of this scene and look ath the same place to see if there is
    # a problem. It seems it misses some colors in the data. Look at picture near 45
