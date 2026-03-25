/// Auto-generated from GBM (100 trees, depth=3, lr=0.1, 43 features)
/// Features used: ap_has_mixed_depth, ap_num_unique_arrays, arith_num_add, arith_num_div, arith_num_mul, arith_num_sub, ci_num_call_ops, ci_num_compare_ops, ci_num_conversion_ops, ci_num_fp_arith_ops, ci_num_int_arith_ops, ci_num_total_ops, ci_ops_per_memory, dep_has_forward_deps, dep_has_loop_carried_deps, dep_num_dependences, dep_safe_for_vectorization, dtype_max_element_size_bytes, dtype_min_element_size_bytes, dtype_num_16bit_accesses, dtype_num_32bit_accesses, dtype_num_64bit_accesses, dtype_num_8bit_accesses, dtype_num_float_accesses, dtype_num_int_accesses, induction_count, lb_final_iv_value, lb_initial_iv_value, lb_step_value, loop_depth, nested_parent_trip_count, num_blocks, num_loads, num_stores, rec_has_recurrence, stride_has_non_affine, stride_max_stride_bytes, stride_min_stride_bytes, stride_num_non_unit_stride, stride_num_unique_base_ptrs, stride_num_unit_stride, trip_count_is_constant, trip_count_value
/// Classes: 0 = NEON, 1 = SSVE
/// Threshold: 0.5

static inline double gbmTree0(const LoopVectorFeatures &F) {
  if (F.trip_count_value <= 31997.000000) {
    if (F.lb_final_iv_value <= 228.000000) {
      if (F.ap_num_unique_arrays <= 6.000000) {
        return -1.0794829616;
      } else {
        return 1.7782448133;
      }
    } else {
      if (F.ap_has_mixed_depth <= 0.500000) {
        return -0.0891968687;
      } else {
        return 2.7035035748;
      }
    }
  } else {
    if (F.stride_num_non_unit_stride <= 0.500000) {
      if (F.dep_has_loop_carried_deps <= 0.500000) {
        return 1.7995791005;
      } else {
        return -1.2618911175;
      }
    } else {
      if (F.arith_num_sub <= 91.500000) {
        return -1.1731280246;
      } else {
        return 4.8183807440;
      }
    }
  }
}

static inline double gbmTree1(const LoopVectorFeatures &F) {
  if (F.trip_count_value <= 15994.000000) {
    if (F.stride_min_stride_bytes <= 768.000000) {
      if (F.ap_has_mixed_depth <= 0.500000) {
        return -0.9997402480;
      } else {
        return 1.8334778557;
      }
    } else {
      if (F.stride_num_unique_base_ptrs <= 2.500000) {
        return -0.1774670600;
      } else {
        return 2.4171862233;
      }
    }
  } else {
    if (F.stride_num_non_unit_stride <= 0.500000) {
      if (F.lb_initial_iv_value <= 6.000000) {
        return 1.2566103609;
      } else {
        return -0.6758084861;
      }
    } else {
      if (F.arith_num_sub <= 91.500000) {
        return -1.1631686486;
      } else {
        return 3.3584116655;
      }
    }
  }
}

static inline double gbmTree2(const LoopVectorFeatures &F) {
  if (F.trip_count_value <= 31997.000000) {
    if (F.lb_final_iv_value <= 96.000000) {
      if (F.stride_num_unique_base_ptrs <= 6.000000) {
        return -1.0547208341;
      } else {
        return 1.6246502395;
      }
    } else {
      if (F.ap_has_mixed_depth <= 0.500000) {
        return -0.1264294470;
      } else {
        return 1.8490612702;
      }
    }
  } else {
    if (F.ci_num_total_ops <= 22.500000) {
      if (F.dep_has_loop_carried_deps <= 0.500000) {
        return 1.3353803130;
      } else {
        return -1.2390843290;
      }
    } else {
      if (F.arith_num_sub <= 91.500000) {
        return -1.0929415100;
      } else {
        return 2.6856431844;
      }
    }
  }
}

static inline double gbmTree3(const LoopVectorFeatures &F) {
  if (F.trip_count_value <= 15994.000000) {
    if (F.stride_min_stride_bytes <= 768.000000) {
      if (F.ap_has_mixed_depth <= 0.500000) {
        return -0.9342418316;
      } else {
        return 1.3019944173;
      }
    } else {
      if (F.ci_num_int_arith_ops <= 1.500000) {
        return 1.2451941881;
      } else {
        return -1.2779998867;
      }
    }
  } else {
    if (F.stride_num_non_unit_stride <= 0.500000) {
      if (F.lb_initial_iv_value <= 6.000000) {
        return 0.9310949454;
      } else {
        return -0.6447321780;
      }
    } else {
      if (F.arith_num_sub <= 91.500000) {
        return -1.1099714733;
      } else {
        return 2.2886341827;
      }
    }
  }
}

static inline double gbmTree4(const LoopVectorFeatures &F) {
  if (F.trip_count_value <= 31997.000000) {
    if (F.lb_final_iv_value <= 96.000000) {
      if (F.ap_num_unique_arrays <= 6.000000) {
        return -0.9944772730;
      } else {
        return 1.4967069029;
      }
    } else {
      if (F.arith_num_div <= 0.500000) {
        return -0.1173433267;
      } else {
        return 2.6509313689;
      }
    }
  } else {
    if (F.ci_num_total_ops <= 22.500000) {
      if (F.dep_safe_for_vectorization <= 0.500000) {
        return -1.2194439884;
      } else {
        return 1.0044961653;
      }
    } else {
      if (F.arith_num_sub <= 91.500000) {
        return -1.0474247335;
      } else {
        return 2.0250275252;
      }
    }
  }
}

static inline double gbmTree5(const LoopVectorFeatures &F) {
  if (F.trip_count_value <= 15994.000000) {
    if (F.stride_min_stride_bytes <= 768.000000) {
      if (F.ap_has_mixed_depth <= 0.500000) {
        return -0.8764461052;
      } else {
        return 1.1558465417;
      }
    } else {
      if (F.stride_num_unique_base_ptrs <= 2.500000) {
        return -0.1928915719;
      } else {
        return 1.8607962180;
      }
    }
  } else {
    if (F.stride_num_non_unit_stride <= 0.500000) {
      if (F.stride_has_non_affine <= 0.500000) {
        return 0.7037893975;
      } else {
        return -1.4352388073;
      }
    } else {
      if (F.arith_num_sub <= 91.500000) {
        return -1.0648887642;
      } else {
        return 1.8371238199;
      }
    }
  }
}

static inline double gbmTree6(const LoopVectorFeatures &F) {
  if (F.trip_count_value <= 31997.000000) {
    if (F.stride_min_stride_bytes <= 768.000000) {
      if (F.trip_count_value <= 15976.000000) {
        return -0.7593746559;
      } else {
        return 0.0419870751;
      }
    } else {
      if (F.ci_num_int_arith_ops <= 1.500000) {
        return 0.9968557327;
      } else {
        return -1.2669125123;
      }
    }
  } else {
    if (F.trip_count_value <= 31999.500000) {
      if (F.dep_safe_for_vectorization <= 0.500000) {
        return -1.2385122170;
      } else {
        return 1.1882970594;
      }
    } else {
      if (F.ci_num_fp_arith_ops <= 0.500000) {
        return 0.4768263546;
      } else {
        return -0.9223868391;
      }
    }
  }
}

static inline double gbmTree7(const LoopVectorFeatures &F) {
  if (F.ci_num_total_ops <= 22.500000) {
    if (F.trip_count_value <= 31998.500000) {
      if (F.ci_num_conversion_ops <= 2.500000) {
        return -0.1669357935;
      } else {
        return 3.9664598683;
      }
    } else {
      if (F.lb_final_iv_value <= 31999.500000) {
        return 1.0271699916;
      } else {
        return 0.2750636157;
      }
    }
  } else {
    if (F.stride_num_unique_base_ptrs <= 6.500000) {
      if (F.dtype_num_int_accesses <= 185.500000) {
        return -1.0229886367;
      } else {
        return -0.1015073828;
      }
    } else {
      if (F.stride_min_stride_bytes <= 6.000000) {
        return 0.4351787956;
      } else {
        return 4.3429262035;
      }
    }
  }
}

static inline double gbmTree8(const LoopVectorFeatures &F) {
  if (F.lb_final_iv_value <= 228.000000) {
    if (F.stride_num_unique_base_ptrs <= 6.000000) {
      if (F.dtype_num_int_accesses <= 200.500000) {
        return -0.8176776546;
      } else {
        return 8.5370101607;
      }
    } else {
      if (F.ci_num_total_ops <= 270.000000) {
        return 3.8140713022;
      } else {
        return 0.5935770449;
      }
    }
  } else {
    if (F.dep_has_loop_carried_deps <= 0.500000) {
      if (F.stride_num_unit_stride <= 4.500000) {
        return 0.1369109705;
      } else {
        return 0.7448035507;
      }
    } else {
      if (F.dtype_num_64bit_accesses <= 4.500000) {
        return -1.2109181760;
      } else {
        return 2.7503284488;
      }
    }
  }
}

static inline double gbmTree9(const LoopVectorFeatures &F) {
  if (F.ci_num_total_ops <= 22.500000) {
    if (F.trip_count_value <= 31998.500000) {
      if (F.ci_num_conversion_ops <= 2.500000) {
        return -0.1466448407;
      } else {
        return 2.9218019541;
      }
    } else {
      if (F.stride_has_non_affine <= 0.500000) {
        return 0.5946718891;
      } else {
        return -1.4425455981;
      }
    }
  } else {
    if (F.ap_num_unique_arrays <= 6.500000) {
      if (F.arith_num_sub <= 92.000000) {
        return -0.8932060316;
      } else {
        return 1.6618484593;
      }
    } else {
      if (F.stride_min_stride_bytes <= 6.000000) {
        return 0.2499681459;
      } else {
        return 2.8422702750;
      }
    }
  }
}

static inline double gbmTree10(const LoopVectorFeatures &F) {
  if (F.lb_final_iv_value <= 228.000000) {
    if (F.stride_num_unique_base_ptrs <= 6.000000) {
      if (F.dtype_num_int_accesses <= 1.000000) {
        return -1.0343936762;
      } else {
        return -0.3953176891;
      }
    } else {
      if (F.arith_num_mul <= 48.500000) {
        return 1.5568330549;
      } else {
        return -1.3399440840;
      }
    }
  } else {
    if (F.dep_has_loop_carried_deps <= 0.500000) {
      if (F.lb_initial_iv_value <= 6.000000) {
        return 0.3574119666;
      } else {
        return -1.2198611288;
      }
    } else {
      if (F.dtype_num_64bit_accesses <= 4.500000) {
        return -1.1886102489;
      } else {
        return 2.3490964803;
      }
    }
  }
}

static inline double gbmTree11(const LoopVectorFeatures &F) {
  if (F.ci_num_total_ops <= 22.500000) {
    if (F.dep_has_loop_carried_deps <= 0.500000) {
      if (F.lb_initial_iv_value <= 6.000000) {
        return 0.3922454054;
      } else {
        return -0.8365217613;
      }
    } else {
      if (F.dtype_num_64bit_accesses <= 4.500000) {
        return -1.1599730227;
      } else {
        return 2.0666527205;
      }
    }
  } else {
    if (F.stride_num_unique_base_ptrs <= 6.500000) {
      if (F.dtype_num_int_accesses <= 185.500000) {
        return -0.9563002207;
      } else {
        return 0.0200302068;
      }
    } else {
      if (F.dtype_min_element_size_bytes <= 6.000000) {
        return 0.1415927112;
      } else {
        return 2.1921993686;
      }
    }
  }
}

static inline double gbmTree12(const LoopVectorFeatures &F) {
  if (F.ci_num_total_ops <= 22.500000) {
    if (F.stride_num_unit_stride <= 4.500000) {
      if (F.arith_num_mul <= 1.500000) {
        return 0.0774331257;
      } else {
        return -1.4336099844;
      }
    } else {
      if (F.stride_num_unit_stride <= 5.500000) {
        return 1.5161215904;
      } else {
        return 0.2923526206;
      }
    }
  } else {
    if (F.arith_num_sub <= 92.000000) {
      if (F.stride_num_unique_base_ptrs <= 6.500000) {
        return -0.8252470921;
      } else {
        return 0.5279254987;
      }
    } else {
      return 1.5397508008;
    }
  }
}

static inline double gbmTree13(const LoopVectorFeatures &F) {
  if (F.trip_count_value <= 31997.000000) {
    if (F.stride_min_stride_bytes <= 768.000000) {
      if (F.ap_has_mixed_depth <= 0.500000) {
        return -0.4667737672;
      } else {
        return 0.9307797510;
      }
    } else {
      if (F.ap_num_unique_arrays <= 2.500000) {
        return -0.1958198879;
      } else {
        return 1.4300020502;
      }
    }
  } else {
    if (F.trip_count_value <= 31999.500000) {
      if (F.num_loads <= 3.500000) {
        return 0.6943967850;
      } else {
        return -1.2217113765;
      }
    } else {
      if (F.ci_num_fp_arith_ops <= 0.500000) {
        return 0.3377220638;
      } else {
        return -0.9490865601;
      }
    }
  }
}

static inline double gbmTree14(const LoopVectorFeatures &F) {
  if (F.trip_count_value <= 15976.000000) {
    if (F.stride_min_stride_bytes <= 768.000000) {
      if (F.ap_has_mixed_depth <= 0.500000) {
        return -0.7161805751;
      } else {
        return 1.0061089472;
      }
    } else {
      if (F.ci_num_int_arith_ops <= 1.500000) {
        return 0.7815090446;
      } else {
        return -1.2598729175;
      }
    }
  } else {
    if (F.stride_num_non_unit_stride <= 0.500000) {
      if (F.arith_num_mul <= 0.500000) {
        return 0.6626880530;
      } else {
        return -0.0021432598;
      }
    } else {
      if (F.arith_num_sub <= 91.500000) {
        return -0.9935057272;
      } else {
        return 1.4473584911;
      }
    }
  }
}

static inline double gbmTree15(const LoopVectorFeatures &F) {
  if (F.trip_count_value <= 15994.000000) {
    if (F.stride_min_stride_bytes <= 768.000000) {
      if (F.ap_has_mixed_depth <= 0.500000) {
        return -0.6694745126;
      } else {
        return 0.8823385678;
      }
    } else {
      if (F.ap_num_unique_arrays <= 2.500000) {
        return -0.2219741003;
      } else {
        return 1.2069221526;
      }
    }
  } else {
    if (F.stride_has_non_affine <= 0.500000) {
      if (F.stride_num_non_unit_stride <= 0.500000) {
        return 0.3312473948;
      } else {
        return -0.8742988146;
      }
    } else {
      if (F.num_loads <= 3.500000) {
        return -1.4176085545;
      } else {
        return -1.1400790679;
      }
    }
  }
}

static inline double gbmTree16(const LoopVectorFeatures &F) {
  if (F.dep_safe_for_vectorization <= 0.500000) {
    if (F.dtype_num_int_accesses <= 200.500000) {
      if (F.dtype_num_int_accesses <= 87.000000) {
        return -1.0891827131;
      } else {
        return -0.1704523632;
      }
    } else {
      return 5.7623512339;
    }
  } else {
    if (F.arith_num_div <= 0.500000) {
      if (F.dtype_num_32bit_accesses <= 2.500000) {
        return -0.2362905269;
      } else {
        return 0.2570385874;
      }
    } else {
      if (F.nested_parent_trip_count <= 350000.000000) {
        return 3.1581779047;
      } else {
        return -1.4348738770;
      }
    }
  }
}

static inline double gbmTree17(const LoopVectorFeatures &F) {
  if (F.stride_has_non_affine <= 0.500000) {
    if (F.trip_count_value <= 31998.500000) {
      if (F.arith_num_mul <= 0.500000) {
        return 0.0248434362;
      } else {
        return -0.7891068811;
      }
    } else {
      if (F.nested_parent_trip_count <= 450000.000000) {
        return 0.2020319687;
      } else {
        return 2.0047066416;
      }
    }
  } else {
    if (F.trip_count_value <= 228.000000) {
      if (F.lb_final_iv_value <= 116.000000) {
        return -1.0672338185;
      } else {
        return -1.1042304063;
      }
    } else {
      if (F.trip_count_value <= 31999.500000) {
        return -1.2141219033;
      } else {
        return -1.3765689976;
      }
    }
  }
}

static inline double gbmTree18(const LoopVectorFeatures &F) {
  if (F.dep_has_loop_carried_deps <= 0.500000) {
    if (F.arith_num_div <= 0.500000) {
      if (F.lb_initial_iv_value <= 6.000000) {
        return 0.1282302638;
      } else {
        return -0.8470455908;
      }
    } else {
      if (F.nested_parent_trip_count <= 350000.000000) {
        return 2.6289181517;
      } else {
        return -1.3745766193;
      }
    }
  } else {
    if (F.dtype_num_8bit_accesses <= 200.000000) {
      if (F.dtype_num_int_accesses <= 87.000000) {
        return -1.0743975655;
      } else {
        return -0.1634589969;
      }
    } else {
      return 3.6698493190;
    }
  }
}

static inline double gbmTree19(const LoopVectorFeatures &F) {
  if (F.stride_max_stride_bytes <= 6.000000) {
    if (F.induction_count <= 1.500000) {
      if (F.ci_num_total_ops <= 9.500000) {
        return 0.6073018610;
      } else {
        return -0.0322390724;
      }
    } else {
      if (F.ci_ops_per_memory <= 1.625000) {
        return 1.9598591896;
      } else {
        return -1.6672697261;
      }
    }
  } else {
    if (F.stride_min_stride_bytes <= 768.000000) {
      if (F.stride_num_unique_base_ptrs <= 6.000000) {
        return -0.8728066823;
      } else {
        return 1.4495061518;
      }
    } else {
      if (F.dtype_num_float_accesses <= 2.000000) {
        return -1.2920899459;
      } else {
        return 0.6457225877;
      }
    }
  }
}

static inline double gbmTree20(const LoopVectorFeatures &F) {
  if (F.trip_count_value <= 15976.000000) {
    if (F.nested_parent_trip_count <= 1024.000000) {
      if (F.stride_num_unique_base_ptrs <= 2.500000) {
        return -0.3877787819;
      } else {
        return 1.2617942201;
      }
    } else {
      if (F.stride_min_stride_bytes <= 3072.000000) {
        return -0.6365584412;
      } else {
        return 0.9506736578;
      }
    }
  } else {
    if (F.lb_final_iv_value <= 31999.500000) {
      if (F.ci_ops_per_memory <= 0.816667) {
        return 1.5114282737;
      } else {
        return 0.2460662015;
      }
    } else {
      if (F.ci_num_fp_arith_ops <= 0.500000) {
        return 0.2417956598;
      } else {
        return -0.8301842576;
      }
    }
  }
}

static inline double gbmTree21(const LoopVectorFeatures &F) {
  if (F.stride_has_non_affine <= 0.500000) {
    if (F.ci_num_total_ops <= 22.500000) {
      if (F.arith_num_div <= 0.500000) {
        return 0.1063981442;
      } else {
        return 1.4861643556;
      }
    } else {
      if (F.ap_num_unique_arrays <= 6.500000) {
        return -0.6409455950;
      } else {
        return 0.7850611841;
      }
    }
  } else {
    if (F.trip_count_value <= 228.000000) {
      if (F.ci_num_int_arith_ops <= 3.500000) {
        return -1.0819537401;
      } else {
        return -1.0516328080;
      }
    } else {
      if (F.lb_final_iv_value <= 31999.500000) {
        return -1.1967543643;
      } else {
        return -1.3339124150;
      }
    }
  }
}

static inline double gbmTree22(const LoopVectorFeatures &F) {
  if (F.stride_has_non_affine <= 0.500000) {
    if (F.dtype_num_float_accesses <= 6.500000) {
      if (F.arith_num_mul <= 0.500000) {
        return 0.2907687033;
      } else {
        return -0.1449261553;
      }
    } else {
      if (F.num_stores <= 4.500000) {
        return -1.4254622314;
      } else {
        return -0.8362274107;
      }
    }
  } else {
    if (F.trip_count_value <= 228.000000) {
      if (F.ci_num_int_arith_ops <= 3.500000) {
        return -1.0735722435;
      } else {
        return -1.0464800843;
      }
    } else {
      if (F.lb_final_iv_value <= 31999.500000) {
        return -1.1756110793;
      } else {
        return -1.2930935536;
      }
    }
  }
}

static inline double gbmTree23(const LoopVectorFeatures &F) {
  if (F.trip_count_value <= 31998.500000) {
    if (F.ci_num_total_ops <= 8.500000) {
      if (F.loop_depth <= 2.500000) {
        return -1.4200988676;
      } else {
        return -1.2948959901;
      }
    } else {
      if (F.arith_num_mul <= 0.500000) {
        return 0.0140554437;
      } else {
        return -0.7424831189;
      }
    }
  } else {
    if (F.nested_parent_trip_count <= 450000.000000) {
      if (F.lb_final_iv_value <= 31999.500000) {
        return 0.4470068733;
      } else {
        return -0.1581451268;
      }
    } else {
      if (F.dtype_max_element_size_bytes <= 3.000000) {
        return -1.6716804827;
      } else {
        return 2.1890599860;
      }
    }
  }
}

static inline double gbmTree24(const LoopVectorFeatures &F) {
  if (F.lb_initial_iv_value <= 6.000000) {
    if (F.ci_num_total_ops <= 11.500000) {
      if (F.stride_min_stride_bytes <= 3.000000) {
        return 0.9676089770;
      } else {
        return 0.1402779989;
      }
    } else {
      if (F.ci_ops_per_memory <= 1.416667) {
        return -0.0133171819;
      } else {
        return -1.2487745193;
      }
    }
  } else {
    if (F.dtype_num_float_accesses <= 1.000000) {
      if (F.nested_parent_trip_count <= 306250.000000) {
        return 2.8836179560;
      } else {
        return -1.1236674005;
      }
    } else {
      if (F.lb_initial_iv_value <= 31998.500000) {
        return -0.9734582904;
      } else {
        return -1.3151406503;
      }
    }
  }
}

static inline double gbmTree25(const LoopVectorFeatures &F) {
  if (F.stride_max_stride_bytes <= 6.000000) {
    if (F.induction_count <= 1.500000) {
      if (F.nested_parent_trip_count <= 75000.000000) {
        return -0.9717141136;
      } else {
        return 0.1494780783;
      }
    } else {
      if (F.ci_ops_per_memory <= 1.625000) {
        return 1.6413739543;
      } else {
        return -1.5765281750;
      }
    }
  } else {
    if (F.stride_min_stride_bytes <= 768.000000) {
      if (F.stride_num_unique_base_ptrs <= 6.000000) {
        return -0.8185015351;
      } else {
        return 1.3824178081;
      }
    } else {
      if (F.dtype_num_float_accesses <= 2.000000) {
        return -1.2714280054;
      } else {
        return 0.5134054243;
      }
    }
  }
}

static inline double gbmTree26(const LoopVectorFeatures &F) {
  if (F.lb_initial_iv_value <= 6.000000) {
    if (F.ci_num_total_ops <= 10.500000) {
      if (F.dtype_num_int_accesses <= 0.500000) {
        return -0.0667045457;
      } else {
        return 1.4720122772;
      }
    } else {
      if (F.ci_ops_per_memory <= 1.416667) {
        return 0.0302120075;
      } else {
        return -1.2812464342;
      }
    }
  } else {
    if (F.dtype_num_float_accesses <= 1.000000) {
      if (F.lb_initial_iv_value <= 8127.000000) {
        return -1.1122855661;
      } else {
        return 2.3939496801;
      }
    } else {
      if (F.trip_count_value <= 31999.500000) {
        return -0.9423384031;
      } else {
        return -1.2749440841;
      }
    }
  }
}

static inline double gbmTree27(const LoopVectorFeatures &F) {
  if (F.trip_count_value <= 15976.000000) {
    if (F.trip_count_value <= 620.000000) {
      if (F.trip_count_value <= 255.500000) {
        return -0.2372855872;
      } else {
        return 0.7208428051;
      }
    } else {
      if (F.arith_num_div <= 0.500000) {
        return -0.9690033844;
      } else {
        return 2.0877224114;
      }
    }
  } else {
    if (F.arith_num_mul <= 0.500000) {
      if (F.dtype_num_float_accesses <= 1.500000) {
        return 0.8692236055;
      } else {
        return 0.0324885834;
      }
    } else {
      if (F.ci_num_total_ops <= 18.500000) {
        return -0.6476595307;
      } else {
        return 0.1048985759;
      }
    }
  }
}

static inline double gbmTree28(const LoopVectorFeatures &F) {
  if (F.dep_has_loop_carried_deps <= 0.500000) {
    if (F.arith_num_div <= 0.500000) {
      if (F.dtype_num_32bit_accesses <= 2.500000) {
        return -0.2072951097;
      } else {
        return 0.1901138674;
      }
    } else {
      if (F.nested_parent_trip_count <= 350000.000000) {
        return 2.0053125580;
      } else {
        return -1.3819808604;
      }
    }
  } else {
    if (F.dtype_num_int_accesses <= 200.500000) {
      if (F.num_stores <= 4.000000) {
        return -1.0590974595;
      } else {
        return -0.4347966948;
      }
    } else {
      return 3.0598234363;
    }
  }
}

static inline double gbmTree29(const LoopVectorFeatures &F) {
  if (F.dtype_num_float_accesses <= 6.500000) {
    if (F.dtype_num_32bit_accesses <= 4.500000) {
      if (F.ci_num_total_ops <= 12.500000) {
        return 0.1729316359;
      } else {
        return -0.4533231204;
      }
    } else {
      if (F.dtype_num_32bit_accesses <= 5.500000) {
        return 1.0752352101;
      } else {
        return 0.0797205481;
      }
    }
  } else {
    if (F.num_stores <= 4.500000) {
      if (F.dep_safe_for_vectorization <= 0.500000) {
        return -1.1055518723;
      } else {
        return -1.4425467138;
      }
    } else {
      if (F.ap_num_unique_arrays <= 6.000000) {
        return -1.0441054617;
      } else {
        return 1.0027730454;
      }
    }
  }
}

static inline double gbmTree30(const LoopVectorFeatures &F) {
  if (F.lb_initial_iv_value <= 6.000000) {
    if (F.ci_num_total_ops <= 12.500000) {
      if (F.stride_num_non_unit_stride <= 3.500000) {
        return 0.2062120758;
      } else {
        return 1.9535103177;
      }
    } else {
      if (F.stride_num_unit_stride <= 4.500000) {
        return -0.4697013394;
      } else {
        return 0.1163513102;
      }
    }
  } else {
    if (F.dtype_num_int_accesses <= 1.000000) {
      if (F.lb_initial_iv_value <= 31998.500000) {
        return -0.9042060909;
      } else {
        return -1.2436438113;
      }
    } else {
      if (F.lb_step_value <= 0.000000) {
        return 1.9659817961;
      } else {
        return -1.0918730431;
      }
    }
  }
}

static inline double gbmTree31(const LoopVectorFeatures &F) {
  if (F.stride_max_stride_bytes <= 6.000000) {
    if (F.induction_count <= 1.500000) {
      if (F.nested_parent_trip_count <= 75000.000000) {
        return -0.9267071235;
      } else {
        return 0.1225720711;
      }
    } else {
      if (F.ap_num_unique_arrays <= 2.500000) {
        return -1.4159046642;
      } else {
        return 1.4930638585;
      }
    }
  } else {
    if (F.trip_count_is_constant <= 0.500000) {
      if (F.ci_num_int_arith_ops <= 1.500000) {
        return 1.7815139688;
      } else {
        return -1.2358267742;
      }
    } else {
      if (F.ci_num_total_ops <= 9.500000) {
        return -0.9934456506;
      } else {
        return -0.3505555895;
      }
    }
  }
}

static inline double gbmTree32(const LoopVectorFeatures &F) {
  if (F.dep_has_loop_carried_deps <= 0.500000) {
    if (F.arith_num_div <= 0.500000) {
      if (F.ci_num_total_ops <= 12.500000) {
        return 0.2251427002;
      } else {
        return -0.0974924181;
      }
    } else {
      if (F.nested_parent_trip_count <= 350000.000000) {
        return 1.9000904165;
      } else {
        return -1.3388355738;
      }
    }
  } else {
    if (F.dtype_num_int_accesses <= 200.500000) {
      if (F.num_stores <= 4.000000) {
        return -1.0387721346;
      } else {
        return -0.3777458725;
      }
    } else {
      return 2.5497256246;
    }
  }
}

static inline double gbmTree33(const LoopVectorFeatures &F) {
  if (F.trip_count_value <= 31998.500000) {
    if (F.ci_num_total_ops <= 8.500000) {
      if (F.stride_max_stride_bytes <= 6.000000) {
        return -1.4740108484;
      } else {
        return -1.2535151411;
      }
    } else {
      if (F.ci_num_total_ops <= 9.500000) {
        return 0.3440318014;
      } else {
        return -0.2232797410;
      }
    }
  } else {
    if (F.nested_parent_trip_count <= 450000.000000) {
      if (F.lb_final_iv_value <= 31999.500000) {
        return 0.3537390868;
      } else {
        return -0.1439247722;
      }
    } else {
      if (F.stride_min_stride_bytes <= 3.000000) {
        return -1.6670454294;
      } else {
        return 1.8588368607;
      }
    }
  }
}

static inline double gbmTree34(const LoopVectorFeatures &F) {
  if (F.lb_initial_iv_value <= 6.000000) {
    if (F.ci_num_total_ops <= 12.500000) {
      if (F.stride_num_non_unit_stride <= 3.500000) {
        return 0.1839729159;
      } else {
        return 1.8140016004;
      }
    } else {
      if (F.stride_num_unit_stride <= 4.500000) {
        return -0.4276425884;
      } else {
        return 0.1042746314;
      }
    }
  } else {
    if (F.dtype_num_float_accesses <= 1.000000) {
      if (F.nested_parent_trip_count <= 306250.000000) {
        return 1.7345625423;
      } else {
        return -1.0808140650;
      }
    } else {
      if (F.dep_num_dependences <= 0.500000) {
        return -1.1306586726;
      } else {
        return -0.3396662282;
      }
    }
  }
}

static inline double gbmTree35(const LoopVectorFeatures &F) {
  if (F.dtype_min_element_size_bytes <= 0.500000) {
    return -1.8838873891;
  } else {
    if (F.ci_num_total_ops <= 7.000000) {
      if (F.stride_max_stride_bytes <= 514.000000) {
        return 1.9188719139;
      } else {
        return -1.2166170779;
      }
    } else {
      if (F.ci_ops_per_memory <= 1.566667) {
        return 0.0209515737;
      } else {
        return -1.2674579182;
      }
    }
  }
}

static inline double gbmTree36(const LoopVectorFeatures &F) {
  if (F.dtype_num_32bit_accesses <= 2.500000) {
    if (F.dtype_num_float_accesses <= 1.500000) {
      if (F.ci_num_total_ops <= 9.500000) {
        return 1.0575079635;
      } else {
        return -0.4081500757;
      }
    } else {
      if (F.trip_count_is_constant <= 0.500000) {
        return 1.4956128862;
      } else {
        return -0.5251211547;
      }
    }
  } else {
    if (F.arith_num_mul <= 0.500000) {
      if (F.trip_count_value <= 23992.000000) {
        return 0.0918984225;
      } else {
        return 0.8092827681;
      }
    } else {
      if (F.lb_final_iv_value <= 31999.500000) {
        return 0.0559789524;
      } else {
        return -0.3920139304;
      }
    }
  }
}

static inline double gbmTree37(const LoopVectorFeatures &F) {
  if (F.dtype_max_element_size_bytes <= 0.500000) {
    return -1.8137813549;
  } else {
    if (F.ci_num_total_ops <= 7.000000) {
      if (F.nested_parent_trip_count <= 200128.000000) {
        return -1.2132020229;
      } else {
        return 1.6829168332;
      }
    } else {
      if (F.ci_ops_per_memory <= 1.566667) {
        return 0.0200528759;
      } else {
        return -1.2316736059;
      }
    }
  }
}

static inline double gbmTree38(const LoopVectorFeatures &F) {
  if (F.num_blocks <= 3.500000) {
    if (F.trip_count_value <= 31998.500000) {
      if (F.ci_num_total_ops <= 8.500000) {
        return -1.3606808951;
      } else {
        return -0.0902226257;
      }
    } else {
      if (F.dtype_num_32bit_accesses <= 2.500000) {
        return -0.2725843164;
      } else {
        return 0.2656991577;
      }
    }
  } else {
    if (F.ci_num_total_ops <= 22.500000) {
      if (F.nested_parent_trip_count <= 75000.000000) {
        return -1.2079525658;
      } else {
        return -1.9164751730;
      }
    } else {
      if (F.dep_num_dependences <= 3.000000) {
        return -0.6361495870;
      } else {
        return -1.2505676546;
      }
    }
  }
}

static inline double gbmTree39(const LoopVectorFeatures &F) {
  if (F.dtype_num_float_accesses <= 5.500000) {
    if (F.dtype_num_float_accesses <= 4.500000) {
      if (F.ci_num_fp_arith_ops <= 1.500000) {
        return 0.0723179071;
      } else {
        return -0.9712221399;
      }
    } else {
      if (F.trip_count_value <= 9992.000000) {
        return -0.9555091665;
      } else {
        return 1.5620739177;
      }
    }
  } else {
    if (F.lb_final_iv_value <= 31999.500000) {
      if (F.trip_count_value <= 31998.500000) {
        return -0.8615900012;
      } else {
        return 0.2285588005;
      }
    } else {
      if (F.nested_parent_trip_count <= 75000.000000) {
        return -1.0535240583;
      } else {
        return -1.5406493212;
      }
    }
  }
}

static inline double gbmTree40(const LoopVectorFeatures &F) {
  if (F.arith_num_mul <= 0.500000) {
    if (F.trip_count_value <= 15976.000000) {
      if (F.ap_has_mixed_depth <= 0.500000) {
        return -0.1823283383;
      } else {
        return 0.8988731620;
      }
    } else {
      if (F.dtype_num_float_accesses <= 1.500000) {
        return 0.6997595564;
      } else {
        return 0.0233487826;
      }
    }
  } else {
    if (F.lb_final_iv_value <= 31999.500000) {
      if (F.lb_final_iv_value <= 31998.500000) {
        return -0.5923459084;
      } else {
        return 0.2184092166;
      }
    } else {
      if (F.dtype_num_int_accesses <= 6.500000) {
        return -0.6547018565;
      } else {
        return 0.6952680526;
      }
    }
  }
}

static inline double gbmTree41(const LoopVectorFeatures &F) {
  if (F.dtype_min_element_size_bytes <= 0.500000) {
    return -1.7135574151;
  } else {
    if (F.ci_num_total_ops <= 7.000000) {
      if (F.stride_min_stride_bytes <= 514.000000) {
        return 1.5493506965;
      } else {
        return -1.1630665033;
      }
    } else {
      if (F.ci_ops_per_memory <= 1.566667) {
        return 0.0181996892;
      } else {
        return -1.1971879226;
      }
    }
  }
}

static inline double gbmTree42(const LoopVectorFeatures &F) {
  if (F.dtype_num_float_accesses <= 5.500000) {
    if (F.dtype_num_float_accesses <= 4.500000) {
      if (F.ci_num_compare_ops <= 1.500000) {
        return 0.0551384306;
      } else {
        return -0.7018097952;
      }
    } else {
      if (F.trip_count_value <= 9992.000000) {
        return -0.9206609212;
      } else {
        return 1.4183021545;
      }
    }
  } else {
    if (F.lb_final_iv_value <= 31999.500000) {
      if (F.trip_count_value <= 31998.500000) {
        return -0.8083768021;
      } else {
        return 0.1832351902;
      }
    } else {
      if (F.nested_parent_trip_count <= 75000.000000) {
        return -1.0481676280;
      } else {
        return -1.4427051567;
      }
    }
  }
}

static inline double gbmTree43(const LoopVectorFeatures &F) {
  if (F.stride_num_unique_base_ptrs <= 6.500000) {
    if (F.dtype_num_float_accesses <= 5.500000) {
      if (F.dtype_num_float_accesses <= 4.500000) {
        return 0.0157380947;
      } else {
        return 0.7212266014;
      }
    } else {
      if (F.lb_final_iv_value <= 31999.500000) {
        return -0.0972017959;
      } else {
        return -1.2780177506;
      }
    }
  } else {
    if (F.dtype_min_element_size_bytes <= 6.000000) {
      if (F.arith_num_add <= 33.500000) {
        return -0.0011459605;
      } else {
        return 2.7039213009;
      }
    } else {
      if (F.ci_ops_per_memory <= 0.136364) {
        return 1.8916942254;
      } else {
        return 2.3632232969;
      }
    }
  }
}

static inline double gbmTree44(const LoopVectorFeatures &F) {
  if (F.lb_initial_iv_value <= 6.000000) {
    if (F.dep_has_loop_carried_deps <= 0.500000) {
      if (F.arith_num_div <= 0.500000) {
        return 0.0714241934;
      } else {
        return 1.2566225727;
      }
    } else {
      if (F.arith_num_add <= 28.500000) {
        return -1.0302266558;
      } else {
        return 0.0098414519;
      }
    }
  } else {
    if (F.dtype_num_float_accesses <= 1.000000) {
      if (F.stride_num_unique_base_ptrs <= 2.500000) {
        return 1.5210077613;
      } else {
        return -1.0661805347;
      }
    } else {
      if (F.dep_num_dependences <= 0.500000) {
        return -1.1124429208;
      } else {
        return -0.2480249829;
      }
    }
  }
}

static inline double gbmTree45(const LoopVectorFeatures &F) {
  if (F.dtype_max_element_size_bytes <= 0.500000) {
    return -1.6098037803;
  } else {
    if (F.ci_num_fp_arith_ops <= 0.500000) {
      if (F.ci_num_total_ops <= 9.500000) {
        return 1.0257788788;
      } else {
        return 0.0061132494;
      }
    } else {
      if (F.trip_count_is_constant <= 0.500000) {
        return 1.3935651390;
      } else {
        return -0.1706659745;
      }
    }
  }
}

static inline double gbmTree46(const LoopVectorFeatures &F) {
  if (F.dtype_num_32bit_accesses <= 2.500000) {
    if (F.ap_num_unique_arrays <= 6.000000) {
      if (F.ci_num_total_ops <= 9.500000) {
        return 0.0890339342;
      } else {
        return -0.4389616359;
      }
    } else {
      if (F.dtype_num_8bit_accesses <= 3.500000) {
        return 1.8105976311;
      } else {
        return -1.2432872850;
      }
    }
  } else {
    if (F.lb_initial_iv_value <= 0.500000) {
      if (F.ci_num_total_ops <= 12.500000) {
        return 0.7809236047;
      } else {
        return 0.0779112905;
      }
    } else {
      if (F.lb_final_iv_value <= 627.500000) {
        return -0.1250978748;
      } else {
        return -1.3337979809;
      }
    }
  }
}

static inline double gbmTree47(const LoopVectorFeatures &F) {
  if (F.arith_num_sub <= 92.000000) {
    if (F.num_blocks <= 3.500000) {
      if (F.dtype_num_32bit_accesses <= 2.500000) {
        return -0.1583114187;
      } else {
        return 0.1039464026;
      }
    } else {
      if (F.ci_num_total_ops <= 22.500000) {
        return -1.4968010503;
      } else {
        return -0.7985887911;
      }
    }
  } else {
    if (F.dtype_num_32bit_accesses <= 187.000000) {
      return 1.6533761989;
    } else {
      return 1.5091006010;
    }
  }
}

static inline double gbmTree48(const LoopVectorFeatures &F) {
  if (F.stride_max_stride_bytes <= 6.000000) {
    if (F.induction_count <= 1.500000) {
      if (F.ci_num_fp_arith_ops <= 0.500000) {
        return 0.2190371351;
      } else {
        return -0.1278211960;
      }
    } else {
      if (F.ci_ops_per_memory <= 1.625000) {
        return 1.3685573721;
      } else {
        return -1.2178202666;
      }
    }
  } else {
    if (F.ci_num_total_ops <= 9.500000) {
      if (F.trip_count_value <= 15994.000000) {
        return -0.6167673827;
      } else {
        return -1.0248669768;
      }
    } else {
      if (F.ci_num_total_ops <= 12.500000) {
        return 0.2118644930;
      } else {
        return -0.8428629649;
      }
    }
  }
}

static inline double gbmTree49(const LoopVectorFeatures &F) {
  if (F.ap_num_unique_arrays <= 6.500000) {
    if (F.arith_num_mul <= 0.500000) {
      if (F.trip_count_value <= 15976.000000) {
        return -0.1302791056;
      } else {
        return 0.2391648092;
      }
    } else {
      if (F.ci_ops_per_memory <= 1.566667) {
        return -0.1040170213;
      } else {
        return -1.2307717017;
      }
    }
  } else {
    if (F.dtype_num_64bit_accesses <= 66.000000) {
      if (F.arith_num_add <= 33.500000) {
        return -0.0469267498;
      } else {
        return 2.0064738158;
      }
    } else {
      if (F.ci_ops_per_memory <= 0.136364) {
        return 1.6753680392;
      } else {
        return 2.0040096861;
      }
    }
  }
}

static inline double gbmTree50(const LoopVectorFeatures &F) {
  if (F.dtype_min_element_size_bytes <= 0.500000) {
    return -1.5397193605;
  } else {
    if (F.lb_initial_iv_value <= 6.000000) {
      if (F.ci_num_total_ops <= 12.500000) {
        return 0.1818438499;
      } else {
        return -0.0571380476;
      }
    } else {
      if (F.dtype_num_int_accesses <= 1.000000) {
        return -0.8395575334;
      } else {
        return 0.7707556539;
      }
    }
  }
}

static inline double gbmTree51(const LoopVectorFeatures &F) {
  if (F.stride_has_non_affine <= 0.500000) {
    if (F.stride_min_stride_bytes <= 3072.000000) {
      if (F.ci_num_fp_arith_ops <= 0.500000) {
        return 0.1550242884;
      } else {
        return -0.1108317985;
      }
    } else {
      if (F.ci_num_fp_arith_ops <= 0.500000) {
        return -1.3324084000;
      } else {
        return 1.5938917262;
      }
    }
  } else {
    if (F.lb_final_iv_value <= 31999.500000) {
      if (F.trip_count_value <= 228.000000) {
        return -1.0277829767;
      } else {
        return -1.1036068769;
      }
    } else {
      if (F.arith_num_add <= 2.500000) {
        return -1.2047066261;
      } else {
        return -1.0519483447;
      }
    }
  }
}

static inline double gbmTree52(const LoopVectorFeatures &F) {
  if (F.stride_num_non_unit_stride <= 0.500000) {
    if (F.ci_num_fp_arith_ops <= 0.500000) {
      if (F.ci_num_total_ops <= 10.500000) {
        return 0.8742938053;
      } else {
        return 0.1552295152;
      }
    } else {
      if (F.trip_count_value <= 31999.500000) {
        return 0.0027740663;
      } else {
        return -0.7943563319;
      }
    }
  } else {
    if (F.trip_count_value <= 2047.500000) {
      if (F.stride_max_stride_bytes <= 3072.000000) {
        return -0.0996709471;
      } else {
        return 0.8408155718;
      }
    } else {
      if (F.arith_num_sub <= 91.500000) {
        return -0.9073306866;
      } else {
        return 1.5063247249;
      }
    }
  }
}

static inline double gbmTree53(const LoopVectorFeatures &F) {
  if (F.ci_num_total_ops <= 5.500000) {
    return -1.5128632164;
  } else {
    if (F.dep_has_loop_carried_deps <= 0.500000) {
      if (F.arith_num_div <= 0.500000) {
        return 0.0280216779;
      } else {
        return 1.1354947776;
      }
    } else {
      if (F.dtype_num_int_accesses <= 200.500000) {
        return -0.5623101585;
      } else {
        return 2.3057538685;
      }
    }
  }
}

static inline double gbmTree54(const LoopVectorFeatures &F) {
  if (F.dtype_num_32bit_accesses <= 2.500000) {
    if (F.dtype_num_32bit_accesses <= 1.500000) {
      if (F.ci_num_total_ops <= 9.500000) {
        return 0.3277746306;
      } else {
        return -0.3109117148;
      }
    } else {
      if (F.lb_final_iv_value <= 24000.000000) {
        return 0.1159627172;
      } else {
        return -0.7526782258;
      }
    }
  } else {
    if (F.arith_num_add <= 1.500000) {
      if (F.ci_ops_per_memory <= 0.664940) {
        return 0.3099641934;
      } else {
        return -1.3275434627;
      }
    } else {
      if (F.trip_count_value <= 31998.500000) {
        return -0.0818788085;
      } else {
        return 0.2774103725;
      }
    }
  }
}

static inline double gbmTree55(const LoopVectorFeatures &F) {
  if (F.num_blocks <= 3.500000) {
    if (F.dtype_num_32bit_accesses <= 2.500000) {
      if (F.stride_num_unique_base_ptrs <= 6.000000) {
        return -0.1585973300;
      } else {
        return 0.9658517749;
      }
    } else {
      if (F.trip_count_value <= 31998.500000) {
        return -0.0946979274;
      } else {
        return 0.2098283662;
      }
    }
  } else {
    if (F.ci_num_total_ops <= 22.500000) {
      if (F.dtype_num_32bit_accesses <= 5.000000) {
        return -1.1541496948;
      } else {
        return -1.5429864298;
      }
    } else {
      if (F.dep_num_dependences <= 3.000000) {
        return -0.5512334495;
      } else {
        return -1.1973672893;
      }
    }
  }
}

static inline double gbmTree56(const LoopVectorFeatures &F) {
  if (F.arith_num_mul <= 0.500000) {
    if (F.stride_num_unique_base_ptrs <= 3.500000) {
      if (F.dtype_num_int_accesses <= 0.500000) {
        return -0.1684002137;
      } else {
        return 0.3539888499;
      }
    } else {
      if (F.ci_num_total_ops <= 18.500000) {
        return 0.7723105243;
      } else {
        return -0.4223620909;
      }
    }
  } else {
    if (F.lb_final_iv_value <= 31999.500000) {
      if (F.lb_final_iv_value <= 31998.500000) {
        return -0.5226821243;
      } else {
        return 0.1637573341;
      }
    } else {
      if (F.dtype_num_int_accesses <= 6.500000) {
        return -0.5702415287;
      } else {
        return 0.6100036190;
      }
    }
  }
}

static inline double gbmTree57(const LoopVectorFeatures &F) {
  if (F.arith_num_mul <= 0.500000) {
    if (F.ap_num_unique_arrays <= 3.500000) {
      if (F.ci_num_compare_ops <= 1.500000) {
        return 0.0413669880;
      } else {
        return -1.4927681860;
      }
    } else {
      if (F.ci_num_total_ops <= 18.500000) {
        return 0.6963020721;
      } else {
        return -0.3886768568;
      }
    }
  } else {
    if (F.ci_ops_per_memory <= 1.566667) {
      if (F.ci_num_total_ops <= 12.500000) {
        return 0.6144055596;
      } else {
        return -0.1346878982;
      }
    } else {
      if (F.lb_final_iv_value <= 31999.500000) {
        return -1.0914100515;
      } else {
        return -1.2241210679;
      }
    }
  }
}

static inline double gbmTree58(const LoopVectorFeatures &F) {
  if (F.stride_num_unique_base_ptrs <= 6.500000) {
    if (F.ci_num_total_ops <= 12.500000) {
      if (F.stride_num_non_unit_stride <= 3.500000) {
        return 0.0763910177;
      } else {
        return 1.6435219204;
      }
    } else {
      if (F.arith_num_add <= 1.500000) {
        return -0.8750285717;
      } else {
        return -0.0063525775;
      }
    }
  } else {
    if (F.arith_num_mul <= 48.500000) {
      if (F.dep_has_forward_deps <= 0.500000) {
        return -1.1952887620;
      } else {
        return 1.5048001518;
      }
    } else {
      if (F.dtype_min_element_size_bytes <= 3.000000) {
        return -1.1834200898;
      } else {
        return -1.2269873254;
      }
    }
  }
}

static inline double gbmTree59(const LoopVectorFeatures &F) {
  if (F.arith_num_sub <= 92.000000) {
    if (F.ap_num_unique_arrays <= 0.500000) {
      return -1.4461148199;
    } else {
      if (F.dtype_num_16bit_accesses <= 2.500000) {
        return 0.0376866730;
      } else {
        return -0.4784329582;
      }
    }
  } else {
    if (F.dtype_min_element_size_bytes <= 3.000000) {
      return 1.5072542544;
    } else {
      return 1.3632176839;
    }
  }
}

static inline double gbmTree60(const LoopVectorFeatures &F) {
  if (F.lb_initial_iv_value <= 6.000000) {
    if (F.ci_num_total_ops <= 12.500000) {
      if (F.stride_num_unit_stride <= 3.500000) {
        return 0.1179083607;
      } else {
        return 1.5968851715;
      }
    } else {
      if (F.ap_has_mixed_depth <= 0.500000) {
        return -0.0821343994;
      } else {
        return 0.7661246087;
      }
    }
  } else {
    if (F.dep_has_forward_deps <= 0.500000) {
      if (F.trip_count_value <= 15976.000000) {
        return -1.0620020812;
      } else {
        return -1.0017997598;
      }
    } else {
      if (F.stride_num_unit_stride <= 3.500000) {
        return 2.9797248953;
      } else {
        return -0.9725096937;
      }
    }
  }
}

static inline double gbmTree61(const LoopVectorFeatures &F) {
  if (F.ap_num_unique_arrays <= 6.500000) {
    if (F.ci_num_call_ops <= 1.500000) {
      if (F.trip_count_value <= 15976.000000) {
        return -0.1643836443;
      } else {
        return 0.0886842884;
      }
    } else {
      if (F.ci_num_call_ops <= 3.500000) {
        return -1.3579175037;
      } else {
        return -1.0299464616;
      }
    }
  } else {
    if (F.dtype_min_element_size_bytes <= 1.500000) {
      if (F.arith_num_add <= 24.500000) {
        return -1.1249155113;
      } else {
        return -1.4337502385;
      }
    } else {
      if (F.arith_num_add <= 33.500000) {
        return 0.6538642747;
      } else {
        return 2.7796404215;
      }
    }
  }
}

static inline double gbmTree62(const LoopVectorFeatures &F) {
  if (F.stride_min_stride_bytes <= 3072.000000) {
    if (F.ci_num_fp_arith_ops <= 0.500000) {
      if (F.ci_num_total_ops <= 9.500000) {
        return 0.6755450319;
      } else {
        return 0.0135345372;
      }
    } else {
      if (F.lb_final_iv_value <= 31999.500000) {
        return 0.0026431287;
      } else {
        return -0.6232401429;
      }
    }
  } else {
    if (F.ci_num_fp_arith_ops <= 0.500000) {
      if (F.dep_safe_for_vectorization <= 0.500000) {
        return -1.0182456833;
      } else {
        return -1.2727821344;
      }
    } else {
      if (F.ci_num_total_ops <= 12.000000) {
        return 1.5439940183;
      } else {
        return -1.0503864362;
      }
    }
  }
}

static inline double gbmTree63(const LoopVectorFeatures &F) {
  if (F.ci_ops_per_memory <= 0.001042) {
    return -1.4216512914;
  } else {
    if (F.stride_has_non_affine <= 0.500000) {
      if (F.lb_initial_iv_value <= 0.500000) {
        return 0.0907794856;
      } else {
        return -0.1430718936;
      }
    } else {
      if (F.lb_final_iv_value <= 10000.000000) {
        return -1.0444223077;
      } else {
        return -1.1477438715;
      }
    }
  }
}

static inline double gbmTree64(const LoopVectorFeatures &F) {
  if (F.arith_num_mul <= 0.500000) {
    if (F.stride_max_stride_bytes <= 6.000000) {
      if (F.ci_num_fp_arith_ops <= 0.500000) {
        return 0.4807466377;
      } else {
        return -0.1286264630;
      }
    } else {
      if (F.dtype_num_float_accesses <= 2.500000) {
        return -0.9189998686;
      } else {
        return 0.1116899381;
      }
    }
  } else {
    if (F.arith_num_sub <= 91.500000) {
      if (F.lb_final_iv_value <= 31999.500000) {
        return -0.0375698285;
      } else {
        return -0.3213306660;
      }
    } else {
      if (F.dtype_num_32bit_accesses <= 187.000000) {
        return 1.4314711072;
      } else {
        return 1.3134356811;
      }
    }
  }
}

static inline double gbmTree65(const LoopVectorFeatures &F) {
  if (F.num_blocks <= 3.500000) {
    if (F.dtype_num_32bit_accesses <= 2.500000) {
      if (F.dtype_num_32bit_accesses <= 1.500000) {
        return -0.0595564821;
      } else {
        return -0.3518787139;
      }
    } else {
      if (F.trip_count_value <= 31998.500000) {
        return -0.1049704456;
      } else {
        return 0.1985122312;
      }
    }
  } else {
    if (F.ci_num_total_ops <= 22.500000) {
      if (F.ci_num_compare_ops <= 2.000000) {
        return -1.1202754299;
      } else {
        return -1.4245033464;
      }
    } else {
      if (F.dep_num_dependences <= 3.000000) {
        return -0.5077315348;
      } else {
        return -1.1758693357;
      }
    }
  }
}

static inline double gbmTree66(const LoopVectorFeatures &F) {
  if (F.arith_num_mul <= 0.500000) {
    if (F.ap_num_unique_arrays <= 3.500000) {
      if (F.ci_num_compare_ops <= 1.500000) {
        return 0.0405638466;
      } else {
        return -1.4450303449;
      }
    } else {
      if (F.ci_num_total_ops <= 18.500000) {
        return 0.6310349715;
      } else {
        return -0.3507606268;
      }
    }
  } else {
    if (F.ci_ops_per_memory <= 1.566667) {
      if (F.ci_num_total_ops <= 12.500000) {
        return 0.5424445645;
      } else {
        return -0.1259884073;
      }
    } else {
      if (F.lb_final_iv_value <= 31999.500000) {
        return -1.0867363216;
      } else {
        return -1.1991371258;
      }
    }
  }
}

static inline double gbmTree67(const LoopVectorFeatures &F) {
  if (F.ap_num_unique_arrays <= 6.500000) {
    if (F.arith_num_add <= 1.500000) {
      if (F.ci_num_total_ops <= 10.500000) {
        return 0.4584785882;
      } else {
        return -0.5049234454;
      }
    } else {
      if (F.ci_num_total_ops <= 8.500000) {
        return -0.7292917709;
      } else {
        return 0.0501845380;
      }
    }
  } else {
    if (F.ci_ops_per_memory <= 0.136364) {
      if (F.ci_num_total_ops <= 270.000000) {
        return 2.3647140712;
      } else {
        return -1.2867923157;
      }
    } else {
      if (F.dtype_max_element_size_bytes <= 6.000000) {
        return -0.1336102182;
      } else {
        return 1.6913634439;
      }
    }
  }
}

static inline double gbmTree68(const LoopVectorFeatures &F) {
  if (F.ci_num_total_ops <= 5.500000) {
    return -1.4010302170;
  } else {
    if (F.ci_num_total_ops <= 12.500000) {
      if (F.stride_num_non_unit_stride <= 3.500000) {
        return 0.0773835178;
      } else {
        return 1.5682394651;
      }
    } else {
      if (F.stride_num_unit_stride <= 4.500000) {
        return -0.3591208060;
      } else {
        return 0.0715774363;
      }
    }
  }
}

static inline double gbmTree69(const LoopVectorFeatures &F) {
  if (F.dtype_num_float_accesses <= 5.500000) {
    if (F.dtype_num_32bit_accesses <= 4.500000) {
      if (F.arith_num_mul <= 1.500000) {
        return 0.0448363635;
      } else {
        return -0.9633859821;
      }
    } else {
      if (F.trip_count_value <= 11192.000000) {
        return -0.2951724316;
      } else {
        return 0.6338627459;
      }
    }
  } else {
    if (F.trip_count_value <= 31999.500000) {
      if (F.ci_ops_per_memory <= 0.015328) {
        return 1.4608250992;
      } else {
        return -0.0812822494;
      }
    } else {
      if (F.ci_num_total_ops <= 22.000000) {
        return -1.2932252782;
      } else {
        return -1.0408326473;
      }
    }
  }
}

static inline double gbmTree70(const LoopVectorFeatures &F) {
  if (F.arith_num_sub <= 92.000000) {
    if (F.stride_min_stride_bytes <= 3072.000000) {
      if (F.stride_num_non_unit_stride <= 0.500000) {
        return 0.0460139980;
      } else {
        return -0.2689997189;
      }
    } else {
      if (F.ci_num_fp_arith_ops <= 0.500000) {
        return -1.2163170502;
      } else {
        return 1.3800396150;
      }
    }
  } else {
    if (F.dtype_num_32bit_accesses <= 187.000000) {
      return 1.4573051767;
    } else {
      return 1.2792245719;
    }
  }
}

static inline double gbmTree71(const LoopVectorFeatures &F) {
  if (F.lb_initial_iv_value <= 3.000000) {
    if (F.lb_initial_iv_value <= 1.500000) {
      if (F.dtype_num_32bit_accesses <= 2.500000) {
        return -0.2031599208;
      } else {
        return 0.0828509999;
      }
    } else {
      if (F.stride_max_stride_bytes <= 6.000000) {
        return 1.2465088536;
      } else {
        return -0.7273901188;
      }
    }
  } else {
    if (F.dtype_min_element_size_bytes <= 3.000000) {
      if (F.stride_num_unique_base_ptrs <= 1.500000) {
        return -1.6506665476;
      } else {
        return 0.6383442400;
      }
    } else {
      if (F.lb_initial_iv_value <= 6.000000) {
        return -1.3251123068;
      } else {
        return -0.4716236506;
      }
    }
  }
}

static inline double gbmTree72(const LoopVectorFeatures &F) {
  if (F.dtype_num_float_accesses <= 5.500000) {
    if (F.dtype_num_64bit_accesses <= 4.500000) {
      if (F.dtype_num_32bit_accesses <= 4.500000) {
        return -0.0208583327;
      } else {
        return 0.3373542698;
      }
    } else {
      return 3.2636782842;
    }
  } else {
    if (F.lb_final_iv_value <= 31999.500000) {
      if (F.trip_count_value <= 31998.500000) {
        return -0.7730789518;
      } else {
        return 0.1256910831;
      }
    } else {
      if (F.nested_parent_trip_count <= 75000.000000) {
        return -1.0263026121;
      } else {
        return -1.2463329618;
      }
    }
  }
}

static inline double gbmTree73(const LoopVectorFeatures &F) {
  if (F.arith_num_mul <= 0.500000) {
    if (F.trip_count_value <= 15976.000000) {
      if (F.trip_count_value <= 620.000000) {
        return 0.0690045317;
      } else {
        return -0.6105477010;
      }
    } else {
      if (F.dtype_num_16bit_accesses <= 2.500000) {
        return 0.2732671133;
      } else {
        return -0.8196671630;
      }
    }
  } else {
    if (F.arith_num_sub <= 91.500000) {
      if (F.stride_max_stride_bytes <= 1.500000) {
        return -1.2348675492;
      } else {
        return -0.0721572922;
      }
    } else {
      if (F.dtype_max_element_size_bytes <= 3.000000) {
        return 1.4042454563;
      } else {
        return 1.2355853789;
      }
    }
  }
}

static inline double gbmTree74(const LoopVectorFeatures &F) {
  if (F.stride_num_unique_base_ptrs <= 6.500000) {
    if (F.dep_num_dependences <= 0.500000) {
      if (F.ci_num_conversion_ops <= 2.500000) {
        return 0.0317657894;
      } else {
        return 1.3085321753;
      }
    } else {
      if (F.ci_num_total_ops <= 11.500000) {
        return 0.7173469067;
      } else {
        return -0.3459018616;
      }
    }
  } else {
    if (F.arith_num_mul <= 48.500000) {
      if (F.rec_has_recurrence <= 0.500000) {
        return -1.1768827844;
      } else {
        return 1.2127237494;
      }
    } else {
      if (F.dtype_num_32bit_accesses <= 66.000000) {
        return -1.1321157947;
      } else {
        return -1.2106928348;
      }
    }
  }
}

static inline double gbmTree75(const LoopVectorFeatures &F) {
  if (F.dtype_max_element_size_bytes <= 0.500000) {
    return -1.3546460529;
  } else {
    if (F.lb_initial_iv_value <= 3.000000) {
      if (F.lb_initial_iv_value <= 1.500000) {
        return 0.0067034047;
      } else {
        return 0.7214120340;
      }
    } else {
      if (F.ci_num_total_ops <= 8.500000) {
        return -1.4633911703;
      } else {
        return -0.1821087326;
      }
    }
  }
}

static inline double gbmTree76(const LoopVectorFeatures &F) {
  if (F.num_blocks <= 3.500000) {
    if (F.dep_has_loop_carried_deps <= 0.500000) {
      if (F.arith_num_div <= 0.500000) {
        return 0.0267198499;
      } else {
        return 1.0541338595;
      }
    } else {
      if (F.trip_count_value <= 11199.500000) {
        return -0.2121654759;
      } else {
        return -1.1275789497;
      }
    }
  } else {
    if (F.ci_num_total_ops <= 22.500000) {
      if (F.ci_num_total_ops <= 21.500000) {
        return -1.1683527110;
      } else {
        return -1.4360950322;
      }
    } else {
      if (F.arith_num_sub <= 1.000000) {
        return -0.6434578299;
      } else {
        return -1.2097678052;
      }
    }
  }
}

static inline double gbmTree77(const LoopVectorFeatures &F) {
  if (F.lb_initial_iv_value <= 6.000000) {
    if (F.ci_num_total_ops <= 12.500000) {
      if (F.stride_num_non_unit_stride <= 3.500000) {
        return 0.1005597560;
      } else {
        return 1.4953015199;
      }
    } else {
      if (F.ap_has_mixed_depth <= 0.500000) {
        return -0.0736686095;
      } else {
        return 0.7091179710;
      }
    }
  } else {
    if (F.dep_has_forward_deps <= 0.500000) {
      if (F.trip_count_value <= 15976.000000) {
        return -1.0514339585;
      } else {
        return -0.9760439254;
      }
    } else {
      if (F.dtype_num_float_accesses <= 3.500000) {
        return 2.5196239365;
      } else {
        return -1.1234403535;
      }
    }
  }
}

static inline double gbmTree78(const LoopVectorFeatures &F) {
  if (F.ap_num_unique_arrays <= 6.500000) {
    if (F.arith_num_add <= 1.500000) {
      if (F.ci_num_total_ops <= 10.500000) {
        return 0.3956180495;
      } else {
        return -0.4816164342;
      }
    } else {
      if (F.ci_num_total_ops <= 8.500000) {
        return -0.6279695004;
      } else {
        return 0.0446906728;
      }
    }
  } else {
    if (F.arith_num_mul <= 48.500000) {
      if (F.dep_has_forward_deps <= 0.500000) {
        return -1.1583982288;
      } else {
        return 1.0832403521;
      }
    } else {
      if (F.dtype_min_element_size_bytes <= 3.000000) {
        return -1.1175775466;
      } else {
        return -1.1859183306;
      }
    }
  }
}

static inline double gbmTree79(const LoopVectorFeatures &F) {
  if (F.dtype_min_element_size_bytes <= 0.500000) {
    return -1.3263418391;
  } else {
    if (F.arith_num_mul <= 0.500000) {
      if (F.lb_final_iv_value <= 15999.500000) {
        return -0.0662621606;
      } else {
        return 0.2062921580;
      }
    } else {
      if (F.arith_num_sub <= 91.500000) {
        return -0.1021734664;
      } else {
        return 1.3280325875;
      }
    }
  }
}

static inline double gbmTree80(const LoopVectorFeatures &F) {
  if (F.stride_min_stride_bytes <= 3072.000000) {
    if (F.ci_num_fp_arith_ops <= 0.500000) {
      if (F.ci_num_total_ops <= 9.500000) {
        return 0.5778886822;
      } else {
        return 0.0220149934;
      }
    } else {
      if (F.ci_num_total_ops <= 8.500000) {
        return -1.3408035768;
      } else {
        return -0.0723199809;
      }
    }
  } else {
    if (F.ci_num_fp_arith_ops <= 0.500000) {
      if (F.arith_num_mul <= 0.500000) {
        return -1.1915283008;
      } else {
        return -1.0112258661;
      }
    } else {
      if (F.dep_safe_for_vectorization <= 0.500000) {
        return -1.0380163697;
      } else {
        return 1.3892843389;
      }
    }
  }
}

static inline double gbmTree81(const LoopVectorFeatures &F) {
  if (F.dtype_num_32bit_accesses <= 2.500000) {
    if (F.dtype_num_32bit_accesses <= 1.500000) {
      if (F.ci_num_total_ops <= 10.500000) {
        return 0.2615630715;
      } else {
        return -0.2640640516;
      }
    } else {
      if (F.lb_final_iv_value <= 24000.000000) {
        return 0.1248122617;
      } else {
        return -0.7617371438;
      }
    }
  } else {
    if (F.trip_count_value <= 31998.500000) {
      if (F.lb_final_iv_value <= 23968.000000) {
        return 0.0154039446;
      } else {
        return -0.9322416741;
      }
    } else {
      if (F.ci_num_total_ops <= 12.500000) {
        return 1.0884756114;
      } else {
        return 0.0582619047;
      }
    }
  }
}

static inline double gbmTree82(const LoopVectorFeatures &F) {
  if (F.lb_initial_iv_value <= 3.000000) {
    if (F.lb_initial_iv_value <= 1.500000) {
      if (F.ci_ops_per_memory <= 1.416667) {
        return 0.0481017496;
      } else {
        return -0.3178177602;
      }
    } else {
      if (F.stride_min_stride_bytes <= 6.000000) {
        return 0.9593539462;
      } else {
        return -0.7705015846;
      }
    }
  } else {
    if (F.dtype_max_element_size_bytes <= 3.000000) {
      if (F.lb_initial_iv_value <= 6.000000) {
        return 0.7044121428;
      } else {
        return -0.9522085755;
      }
    } else {
      if (F.lb_initial_iv_value <= 6.000000) {
        return -1.2996451639;
      } else {
        return -0.4451537661;
      }
    }
  }
}

static inline double gbmTree83(const LoopVectorFeatures &F) {
  if (F.ci_ops_per_memory <= 0.001042) {
    return -1.3123319563;
  } else {
    if (F.stride_has_non_affine <= 0.500000) {
      if (F.lb_initial_iv_value <= 0.500000) {
        return 0.0790009485;
      } else {
        return -0.1262529595;
      }
    } else {
      if (F.lb_final_iv_value <= 10000.000000) {
        return -1.0342109586;
      } else {
        return -1.1323447396;
      }
    }
  }
}

static inline double gbmTree84(const LoopVectorFeatures &F) {
  if (F.dtype_num_int_accesses <= 185.500000) {
    if (F.dep_safe_for_vectorization <= 0.500000) {
      if (F.trip_count_value <= 11199.500000) {
        return -0.4354774392;
      } else {
        return -1.1141648166;
      }
    } else {
      if (F.arith_num_div <= 0.500000) {
        return 0.0080647810;
      } else {
        return 0.9667868437;
      }
    }
  } else {
    if (F.ci_num_total_ops <= 279.000000) {
      if (F.dtype_max_element_size_bytes <= 3.000000) {
        return -1.0681239071;
      } else {
        return 7.7829584267;
      }
    } else {
      if (F.arith_num_sub <= 15.500000) {
        return -1.0699567138;
      } else {
        return 1.3268390628;
      }
    }
  }
}

static inline double gbmTree85(const LoopVectorFeatures &F) {
  if (F.num_blocks <= 3.500000) {
    if (F.dtype_num_16bit_accesses <= 2.500000) {
      if (F.dtype_num_16bit_accesses <= 1.000000) {
        return 0.0019561866;
      } else {
        return 0.6893417365;
      }
    } else {
      if (F.ci_num_total_ops <= 13.500000) {
        return -0.7633713770;
      } else {
        return 0.1006185034;
      }
    }
  } else {
    if (F.ci_num_total_ops <= 22.500000) {
      if (F.dtype_num_int_accesses <= 5.000000) {
        return -1.1464501710;
      } else {
        return -1.3809545122;
      }
    } else {
      if (F.arith_num_sub <= 1.000000) {
        return -0.5992743865;
      } else {
        return -1.1875059438;
      }
    }
  }
}

static inline double gbmTree86(const LoopVectorFeatures &F) {
  if (F.stride_min_stride_bytes <= 3072.000000) {
    if (F.stride_max_stride_bytes <= 3072.000000) {
      if (F.ci_num_fp_arith_ops <= 0.500000) {
        return 0.1260820292;
      } else {
        return -0.0925733368;
      }
    } else {
      if (F.stride_num_non_unit_stride <= 1.500000) {
        return -1.1070548206;
      } else {
        return 2.9650075675;
      }
    }
  } else {
    if (F.ci_num_fp_arith_ops <= 0.500000) {
      if (F.arith_num_mul <= 0.500000) {
        return -1.1590579004;
      } else {
        return -1.0083494711;
      }
    } else {
      if (F.ap_num_unique_arrays <= 2.500000) {
        return -1.0307235069;
      } else {
        return 1.3578377204;
      }
    }
  }
}

static inline double gbmTree87(const LoopVectorFeatures &F) {
  if (F.lb_initial_iv_value <= 3.000000) {
    if (F.lb_initial_iv_value <= 1.500000) {
      if (F.ci_ops_per_memory <= 1.416667) {
        return 0.0447045830;
      } else {
        return -0.2979687672;
      }
    } else {
      if (F.stride_max_stride_bytes <= 6.000000) {
        return 0.8428948522;
      } else {
        return -0.7076499211;
      }
    }
  } else {
    if (F.stride_num_unique_base_ptrs <= 1.500000) {
      if (F.ci_num_int_arith_ops <= 1.500000) {
        return -1.1754092605;
      } else {
        return -1.6865197184;
      }
    } else {
      if (F.stride_max_stride_bytes <= 3.000000) {
        return 0.5232092516;
      } else {
        return -0.7834297476;
      }
    }
  }
}

static inline double gbmTree88(const LoopVectorFeatures &F) {
  if (F.arith_num_mul <= 0.500000) {
    if (F.ap_num_unique_arrays <= 3.500000) {
      if (F.ci_num_compare_ops <= 1.500000) {
        return 0.0344341685;
      } else {
        return -1.3709258354;
      }
    } else {
      if (F.trip_count_value <= 9992.000000) {
        return -0.0686574450;
      } else {
        return 0.6727341892;
      }
    }
  } else {
    if (F.ci_ops_per_memory <= 1.416667) {
      if (F.dep_num_dependences <= 0.500000) {
        return 0.0240132300;
      } else {
        return -0.8233460640;
      }
    } else {
      if (F.ci_num_fp_arith_ops <= 0.500000) {
        return -1.0388635637;
      } else {
        return 0.6110338925;
      }
    }
  }
}

static inline double gbmTree89(const LoopVectorFeatures &F) {
  if (F.stride_num_unique_base_ptrs <= 6.500000) {
    if (F.arith_num_add <= 1.500000) {
      if (F.ci_ops_per_memory <= 0.661503) {
        return -0.0448634843;
      } else {
        return -0.5658449810;
      }
    } else {
      if (F.arith_num_div <= 0.500000) {
        return 0.0066665921;
      } else {
        return 0.9050797218;
      }
    }
  } else {
    if (F.stride_max_stride_bytes <= 1.500000) {
      if (F.arith_num_add <= 24.500000) {
        return -1.0648464880;
      } else {
        return -1.3346183988;
      }
    } else {
      if (F.arith_num_add <= 33.500000) {
        return 0.4632351416;
      } else {
        return 2.2440485547;
      }
    }
  }
}

static inline double gbmTree90(const LoopVectorFeatures &F) {
  if (F.dtype_num_float_accesses <= 5.500000) {
    if (F.stride_num_unit_stride <= 4.500000) {
      if (F.arith_num_mul <= 1.500000) {
        return 0.0303561052;
      } else {
        return -0.9925608389;
      }
    } else {
      if (F.trip_count_value <= 9992.000000) {
        return 0.0076656369;
      } else {
        return 0.5228050596;
      }
    }
  } else {
    if (F.trip_count_value <= 31999.500000) {
      if (F.trip_count_value <= 31998.500000) {
        return -0.7336580935;
      } else {
        return 0.1150311656;
      }
    } else {
      if (F.ci_num_total_ops <= 22.000000) {
        return -1.2313351139;
      } else {
        return -1.0342620471;
      }
    }
  }
}

static inline double gbmTree91(const LoopVectorFeatures &F) {
  if (F.stride_num_unique_base_ptrs <= 6.500000) {
    if (F.dep_num_dependences <= 0.500000) {
      if (F.ci_num_fp_arith_ops <= 0.500000) {
        return 0.1977335440;
      } else {
        return -0.0667729447;
      }
    } else {
      if (F.ci_ops_per_memory <= 1.104762) {
        return -0.0674827952;
      } else {
        return -0.7768105904;
      }
    }
  } else {
    if (F.dtype_max_element_size_bytes <= 6.000000) {
      if (F.ci_num_fp_arith_ops <= 24.500000) {
        return 0.8390140269;
      } else {
        return -1.3317728378;
      }
    } else {
      if (F.ci_num_total_ops <= 285.000000) {
        return 1.2837272543;
      } else {
        return 1.5733862143;
      }
    }
  }
}

static inline double gbmTree92(const LoopVectorFeatures &F) {
  if (F.ci_num_compare_ops <= 1.500000) {
    if (F.trip_count_value <= 31998.500000) {
      if (F.ci_num_total_ops <= 8.500000) {
        return -1.3464796173;
      } else {
        return -0.0300876414;
      }
    } else {
      if (F.nested_parent_trip_count <= 450000.000000) {
        return 0.0512951053;
      } else {
        return 1.2469825030;
      }
    }
  } else {
    if (F.ci_num_total_ops <= 17.500000) {
      if (F.nested_parent_trip_count <= 300000.000000) {
        return -1.2212561650;
      } else {
        return -1.4821682204;
      }
    } else {
      if (F.dtype_num_32bit_accesses <= 5.500000) {
        return 1.4990458735;
      } else {
        return -0.9925690467;
      }
    }
  }
}

static inline double gbmTree93(const LoopVectorFeatures &F) {
  if (F.ci_ops_per_memory <= 0.001042) {
    return -1.2865203866;
  } else {
    if (F.stride_min_stride_bytes <= 3072.000000) {
      if (F.stride_max_stride_bytes <= 3072.000000) {
        return 0.0077730391;
      } else {
        return -0.8619414344;
      }
    } else {
      if (F.ci_num_fp_arith_ops <= 0.500000) {
        return -1.1416961824;
      } else {
        return 1.2625151534;
      }
    }
  }
}

static inline double gbmTree94(const LoopVectorFeatures &F) {
  if (F.stride_num_unique_base_ptrs <= 6.500000) {
    if (F.ci_num_call_ops <= 1.500000) {
      if (F.arith_num_mul <= 0.500000) {
        return 0.0829678565;
      } else {
        return -0.0890323215;
      }
    } else {
      if (F.ci_num_total_ops <= 20.500000) {
        return -1.4411466407;
      } else {
        return -1.0323491346;
      }
    }
  } else {
    if (F.dtype_max_element_size_bytes <= 6.000000) {
      if (F.ci_num_fp_arith_ops <= 24.500000) {
        return 0.7409035083;
      } else {
        return -1.2913856824;
      }
    } else {
      if (F.ci_num_total_ops <= 285.000000) {
        return 1.2501028265;
      } else {
        return 1.4916135850;
      }
    }
  }
}

static inline double gbmTree95(const LoopVectorFeatures &F) {
  if (F.dtype_num_16bit_accesses <= 2.500000) {
    if (F.dtype_num_16bit_accesses <= 1.000000) {
      if (F.lb_initial_iv_value <= 0.500000) {
        return 0.0807462392;
      } else {
        return -0.2610478443;
      }
    } else {
      if (F.trip_count_value <= 15994.000000) {
        return -1.1488896391;
      } else {
        return 0.9465565370;
      }
    }
  } else {
    if (F.ci_num_total_ops <= 13.500000) {
      if (F.nested_parent_trip_count <= 350000.000000) {
        return -0.5765337935;
      } else {
        return -1.3685532590;
      }
    } else {
      if (F.num_stores <= 1.500000) {
        return 1.4028448860;
      } else {
        return -0.1862860284;
      }
    }
  }
}

static inline double gbmTree96(const LoopVectorFeatures &F) {
  if (F.dtype_num_int_accesses <= 185.500000) {
    if (F.dep_safe_for_vectorization <= 0.500000) {
      if (F.trip_count_value <= 11199.500000) {
        return -0.3839772853;
      } else {
        return -1.1041637009;
      }
    } else {
      if (F.arith_num_div <= 0.500000) {
        return 0.0071992724;
      } else {
        return 0.8318405737;
      }
    }
  } else {
    if (F.ci_num_total_ops <= 383.500000) {
      if (F.dtype_num_32bit_accesses <= 194.000000) {
        return 1.6674202643;
      } else {
        return 3.9143475373;
      }
    } else {
      if (F.arith_num_sub <= 62.500000) {
        return -0.7619137752;
      } else {
        return 2.2021360935;
      }
    }
  }
}

static inline double gbmTree97(const LoopVectorFeatures &F) {
  if (F.stride_num_unit_stride <= 200.000000) {
    if (F.stride_min_stride_bytes <= 768.000000) {
      if (F.dtype_num_float_accesses <= 1.500000) {
        return 0.1129484438;
      } else {
        return -0.1198590177;
      }
    } else {
      if (F.dtype_num_float_accesses <= 2.000000) {
        return -1.1897641101;
      } else {
        return 0.3838917181;
      }
    }
  } else {
    if (F.stride_max_stride_bytes <= 1.500000) {
      return 2.0735619806;
    } else {
      return -1.0136001062;
    }
  }
}

static inline double gbmTree98(const LoopVectorFeatures &F) {
  if (F.stride_num_unique_base_ptrs <= 0.500000) {
    return -1.2591854512;
  } else {
    if (F.ci_num_compare_ops <= 1.500000) {
      if (F.trip_count_value <= 31998.500000) {
        return -0.0574194626;
      } else {
        return 0.0867412834;
      }
    } else {
      if (F.ci_num_total_ops <= 17.500000) {
        return -1.3199917753;
      } else {
        return 0.1114335250;
      }
    }
  }
}

static inline double gbmTree99(const LoopVectorFeatures &F) {
  if (F.arith_num_mul <= 1.500000) {
    if (F.dtype_num_32bit_accesses <= 2.500000) {
      if (F.trip_count_value <= 31998.500000) {
        return 0.0034750401;
      } else {
        return -0.3038760417;
      }
    } else {
      if (F.trip_count_value <= 23992.000000) {
        return -0.0069080914;
      } else {
        return 0.3760543859;
      }
    }
  } else {
    if (F.ci_num_total_ops <= 16.000000) {
      if (F.trip_count_value <= 31997.000000) {
        return -1.1236896125;
      } else {
        return -1.3230588200;
      }
    } else {
      if (F.ci_num_total_ops <= 21.500000) {
        return 0.0726273375;
      } else {
        return -0.6816543766;
      }
    }
  }
}

static bool shouldUseStreamingSVE(const LoopVectorFeatures &F) {
  double raw = -1.3398264437;
  raw += 0.1 * gbmTree0(F);
  raw += 0.1 * gbmTree1(F);
  raw += 0.1 * gbmTree2(F);
  raw += 0.1 * gbmTree3(F);
  raw += 0.1 * gbmTree4(F);
  raw += 0.1 * gbmTree5(F);
  raw += 0.1 * gbmTree6(F);
  raw += 0.1 * gbmTree7(F);
  raw += 0.1 * gbmTree8(F);
  raw += 0.1 * gbmTree9(F);
  raw += 0.1 * gbmTree10(F);
  raw += 0.1 * gbmTree11(F);
  raw += 0.1 * gbmTree12(F);
  raw += 0.1 * gbmTree13(F);
  raw += 0.1 * gbmTree14(F);
  raw += 0.1 * gbmTree15(F);
  raw += 0.1 * gbmTree16(F);
  raw += 0.1 * gbmTree17(F);
  raw += 0.1 * gbmTree18(F);
  raw += 0.1 * gbmTree19(F);
  raw += 0.1 * gbmTree20(F);
  raw += 0.1 * gbmTree21(F);
  raw += 0.1 * gbmTree22(F);
  raw += 0.1 * gbmTree23(F);
  raw += 0.1 * gbmTree24(F);
  raw += 0.1 * gbmTree25(F);
  raw += 0.1 * gbmTree26(F);
  raw += 0.1 * gbmTree27(F);
  raw += 0.1 * gbmTree28(F);
  raw += 0.1 * gbmTree29(F);
  raw += 0.1 * gbmTree30(F);
  raw += 0.1 * gbmTree31(F);
  raw += 0.1 * gbmTree32(F);
  raw += 0.1 * gbmTree33(F);
  raw += 0.1 * gbmTree34(F);
  raw += 0.1 * gbmTree35(F);
  raw += 0.1 * gbmTree36(F);
  raw += 0.1 * gbmTree37(F);
  raw += 0.1 * gbmTree38(F);
  raw += 0.1 * gbmTree39(F);
  raw += 0.1 * gbmTree40(F);
  raw += 0.1 * gbmTree41(F);
  raw += 0.1 * gbmTree42(F);
  raw += 0.1 * gbmTree43(F);
  raw += 0.1 * gbmTree44(F);
  raw += 0.1 * gbmTree45(F);
  raw += 0.1 * gbmTree46(F);
  raw += 0.1 * gbmTree47(F);
  raw += 0.1 * gbmTree48(F);
  raw += 0.1 * gbmTree49(F);
  raw += 0.1 * gbmTree50(F);
  raw += 0.1 * gbmTree51(F);
  raw += 0.1 * gbmTree52(F);
  raw += 0.1 * gbmTree53(F);
  raw += 0.1 * gbmTree54(F);
  raw += 0.1 * gbmTree55(F);
  raw += 0.1 * gbmTree56(F);
  raw += 0.1 * gbmTree57(F);
  raw += 0.1 * gbmTree58(F);
  raw += 0.1 * gbmTree59(F);
  raw += 0.1 * gbmTree60(F);
  raw += 0.1 * gbmTree61(F);
  raw += 0.1 * gbmTree62(F);
  raw += 0.1 * gbmTree63(F);
  raw += 0.1 * gbmTree64(F);
  raw += 0.1 * gbmTree65(F);
  raw += 0.1 * gbmTree66(F);
  raw += 0.1 * gbmTree67(F);
  raw += 0.1 * gbmTree68(F);
  raw += 0.1 * gbmTree69(F);
  raw += 0.1 * gbmTree70(F);
  raw += 0.1 * gbmTree71(F);
  raw += 0.1 * gbmTree72(F);
  raw += 0.1 * gbmTree73(F);
  raw += 0.1 * gbmTree74(F);
  raw += 0.1 * gbmTree75(F);
  raw += 0.1 * gbmTree76(F);
  raw += 0.1 * gbmTree77(F);
  raw += 0.1 * gbmTree78(F);
  raw += 0.1 * gbmTree79(F);
  raw += 0.1 * gbmTree80(F);
  raw += 0.1 * gbmTree81(F);
  raw += 0.1 * gbmTree82(F);
  raw += 0.1 * gbmTree83(F);
  raw += 0.1 * gbmTree84(F);
  raw += 0.1 * gbmTree85(F);
  raw += 0.1 * gbmTree86(F);
  raw += 0.1 * gbmTree87(F);
  raw += 0.1 * gbmTree88(F);
  raw += 0.1 * gbmTree89(F);
  raw += 0.1 * gbmTree90(F);
  raw += 0.1 * gbmTree91(F);
  raw += 0.1 * gbmTree92(F);
  raw += 0.1 * gbmTree93(F);
  raw += 0.1 * gbmTree94(F);
  raw += 0.1 * gbmTree95(F);
  raw += 0.1 * gbmTree96(F);
  raw += 0.1 * gbmTree97(F);
  raw += 0.1 * gbmTree98(F);
  raw += 0.1 * gbmTree99(F);
  // sigmoid
  double prob = 1.0 / (1.0 + exp(-raw));
  return prob >= 0.5;
}
