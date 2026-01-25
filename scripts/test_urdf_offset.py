#!/usr/bin/env python3
"""
Standalone diagnostic script to check for inherent offset between two URDF files.
Usage: python test_urdf_offset.py
"""

import os
import sys
from pathlib import Path
import jax
import jax.numpy as jnp
import jaxlie
from jaxmp import JaxKinTree
from jaxmp.extras.urdf_loader import load_urdf
import numpy as onp

# Configure JAX
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.25"
jax.config.update("jax_platform_name", "cpu")

def test_urdf_offset():
    """
    Test if there's an inherent position offset between the two URDFs.
    """
    
    # ============ CONFIGURE PATHS HERE ============
    urdf_yam_path = Path("/mnt/spare-ssd/hudsonssd/development/real2render2real/data/yam_description_new/urdf/yam.urdf")
    urdf_i2rt_path = Path("/mnt/spare-ssd/hudsonssd/Downloads/i2rt_yam/modified_i2rt_yam.urdf")
    
    ee_link_name = "gripper"
    # ============================================
    
    if not urdf_yam_path.exists():
        print(f"❌ Error: yam.urdf not found at: {urdf_yam_path}")
        return None
    
    if not urdf_i2rt_path.exists():
        print(f"❌ Error: modified_i2rt_yam.urdf not found at: {urdf_i2rt_path}")
        return None
    
    print(f"\n📂 Loading URDFs...")
    print(f"  yam.urdf: {urdf_yam_path}")
    print(f"  i2rt.urdf: {urdf_i2rt_path}")
    
    try:
        print("\n🔄 Loading URDFs with jaxmp...")
        urdf_yam = load_urdf(None, urdf_yam_path)
        urdf_i2rt = load_urdf(None, urdf_i2rt_path)
        
        kin_yam = JaxKinTree.from_urdf(urdf_yam)
        kin_i2rt = JaxKinTree.from_urdf(urdf_i2rt)
        
        print(f"✅ URDFs loaded successfully")
        print(f"  yam.urdf has {kin_yam.num_actuated_joints} actuated joints")
        print(f"  i2rt.urdf has {kin_i2rt.num_actuated_joints} actuated joints")
        
    except Exception as e:
        print(f"❌ Error loading URDFs: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    print(f"\n🔍 Joint names:")
    print(f"  yam.urdf: {kin_yam.joint_names}")
    print(f"  i2rt.urdf: {kin_i2rt.joint_names}")
    
    try:
        ee_idx_yam = kin_yam.joint_names.index(ee_link_name)
        ee_idx_i2rt = kin_i2rt.joint_names.index(ee_link_name)
        print(f"\n✅ Joint '{ee_link_name}' found at index {ee_idx_yam} (yam) and {ee_idx_i2rt} (i2rt)")
    except ValueError:
        print(f"\n❌ Error: Joint '{ee_link_name}' not found")
        return None
    
    # Use the correct number of actuated joints (7, not 8)
    # gripper_mirror is a mimic joint, so only 7 are actuated
    num_joints = kin_yam.num_actuated_joints
    print(f"\n📝 Using {num_joints} actuated joints (gripper_mirror is mimic)")
    
    test_configs = [
        jnp.zeros(num_joints),  # All zeros
        jnp.array([0.07, 0.12, 0.42, 0.17, 0.0, 0.0, 0.045]),  # 7 joints (no gripper_mirror)
        jnp.array([0.5, 0.8, 1.0, -0.5, 0.3, 0.5, 0.04]),  # 7 joints
    ]
    
    print("\n" + "="*80)
    print("URDF OFFSET DIAGNOSTIC TEST")
    print("="*80)
    
    max_pos_diff = 0.0
    max_quat_error = 0.0
    avg_offset = onp.zeros(3)
    
    for i, joint_config in enumerate(test_configs):
        # Compute FK
        fk_yam = kin_yam.forward_kinematics(joint_config)
        fk_i2rt = kin_i2rt.forward_kinematics(joint_config)
        
        # Get EE poses
        ee_pose_yam = fk_yam[ee_idx_yam]
        ee_pose_i2rt = fk_i2rt[ee_idx_i2rt]
        
        # Convert to SE3
        se3_yam = jaxlie.SE3(ee_pose_yam)
        se3_i2rt = jaxlie.SE3(ee_pose_i2rt)
        
        # Extract positions
        pos_yam = onp.array(se3_yam.translation())
        pos_i2rt = onp.array(se3_i2rt.translation())
        
        # Calculate differences
        pos_diff_vec = pos_yam - pos_i2rt
        pos_diff = onp.linalg.norm(pos_diff_vec)
        max_pos_diff = max(max_pos_diff, pos_diff)
        avg_offset += pos_diff_vec
        
        # Orientation difference
        quat_yam = onp.array(se3_yam.rotation().wxyz)
        quat_i2rt = onp.array(se3_i2rt.rotation().wxyz)
        quat_error = 2 * onp.arccos(onp.abs(onp.dot(quat_yam, quat_i2rt)).clip(-1, 1))
        max_quat_error = max(max_quat_error, quat_error)
        
        print(f"\n📍 Config {i+1}: [{', '.join([f'{x:.3f}' for x in joint_config])}]")
        print(f"  yam    EE: [{pos_yam[0]:+.6f}, {pos_yam[1]:+.6f}, {pos_yam[2]:+.6f}]")
        print(f"  i2rt   EE: [{pos_i2rt[0]:+.6f}, {pos_i2rt[1]:+.6f}, {pos_i2rt[2]:+.6f}]")
        print(f"  Δ Vector:  [{pos_diff_vec[0]:+.6f}, {pos_diff_vec[1]:+.6f}, {pos_diff_vec[2]:+.6f}]")
        print(f"  Δ Magnitude: {pos_diff:.6f} m ({pos_diff*1000:.2f} mm)")
        print(f"  Δ Orientation: {quat_error:.4f} rad ({onp.degrees(quat_error):.1f}°)")
    
    avg_offset /= len(test_configs)
    
    print("\n" + "="*80)
    print("📊 DIAGNOSTIC RESULTS:")
    print("="*80)
    
    if max_pos_diff < 0.001:
        print("\n✅ URDFs MATCH - No inherent position offset")
        print(f"   Max position difference: {max_pos_diff*1000:.2f} mm")
        print("\n❌ Your 0.0725m error is from MISMATCHED LINKS in error calculation")
        print("\n🔧 FIX: In yam_coffee_maker.py:")
        print("   FROM: body_names=['link_6']")
        print("   TO:   body_names=['link_left_finger']")
    else:
        print(f"\n⚠️ URDFs DON'T MATCH - Inherent offset exists!")
        print(f"   Max position difference: {max_pos_diff*1000:.2f} mm")
        print(f"   Average position offset: [{avg_offset[0]:.6f}, {avg_offset[1]:.6f}, {avg_offset[2]:.6f}]")
        print(f"   Max orientation difference: {onp.degrees(max_quat_error):.1f}°")
        
        print(f"\n💡 Analysis:")
        print(f"   - Your observed error: 72.5mm")
        print(f"   - URDF inherent offset: {max_pos_diff*1000:.1f}mm")
        print(f"   - Orientation offset: ~90° (π/2 radians) ← Already compensated")
        
        if max_pos_diff > 0.040:  # > 40mm
            print(f"\n   ⚠️ The {max_pos_diff*1000:.1f}mm offset is LARGE and explains most of your error!")
            print(f"   This is likely due to different gripper joint origins in the URDFs.")
        
        print(f"\n🔧 TWO FIXES NEEDED:")
        print(f"   1. Change body_names=['link_6'] to body_names=['link_left_finger']")
        print(f"   2. Apply position offset correction:")
        print(f"      POSITION_OFFSET = np.array([{avg_offset[0]:.6f}, {avg_offset[1]:.6f}, {avg_offset[2]:.6f}])")
    
    print("="*80 + "\n")
    return max_pos_diff


if __name__ == "__main__":
    print("\n🤖 URDF Offset Diagnostic Tool\n")
    
    try:
        offset = test_urdf_offset()
        sys.exit(0 if offset is not None else 1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)