FILE(REMOVE_RECURSE
  "../srv_gen"
  "../srv_gen"
  "../src/cob_vision_ptam_monoslam/srv"
  "CMakeFiles/ROSBUILD_gensrv_py"
  "../src/cob_vision_ptam_monoslam/srv/__init__.py"
  "../src/cob_vision_ptam_monoslam/srv/_point_cloud_server.py"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/ROSBUILD_gensrv_py.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)
