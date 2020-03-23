def orientation_evaluation(gt_pose, pred_rotmat, batch_size, curr_batch_size, step):

    import torch
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    
    # Orientation evaluation
    # Taking as input gt_pose in axis-angle representation and pred_rotmat in rotation matrix representation

    gt_rotvec = torch.zeros((curr_batch_size,24,3), dtype=torch.double) # Reshaping the axis-angle (batch, 72) to (batch, 24, 3) for rotation vector compatibility

    for i, row in enumerate(gt_pose):
        gt_rotvec[i] = torch.reshape(row,(24, -1))

    #print("gt_rotvec", gt_rotvec.shape, gt_rotvec)

    # Get prediction as rotation vectors

    pred_rotvec_arr = np.zeros((curr_batch_size,24,3)) # Has to be a numpy array because it works with Rotation

    for i, row in enumerate(pred_rotmat):
        r = R.from_dcm(row.cpu()) # create the rotation object from the rotation matrix
        pred_rotvec_arr[i] = R.as_rotvec(r) # write it as rotation vectors in pred_rotvec_arr

    pred_rotvec = torch.from_numpy(pred_rotvec_arr) # transform it to a tensor

    #print("pred_rotvec", pred_rotvec.shape, pred_rotvec)

    orientation_error_per_part = np.degrees(torch.sqrt((gt_rotvec - pred_rotvec)**2))
    # This gives the error per part

    #print("error per part", orientation_error_non_reduced.shape, orientation_error_non_reduced)

    orientation_error = np.degrees(torch.sqrt((gt_rotvec - pred_rotvec)**2).sum(dim=-1).mean(dim=-1))
    # The reduction above is wrong. For a 90 degree error in one angle, it averages out 3.75 degrees, which
    # is 90/24. The correct reduction would be a mean of 1.25 (90/72), because there are 72 angles (3 for each part)
    # To remove the root, add [:,1:,:] to gt_euler and pred_euler above

    orientation_error_new = np.degrees(torch.sqrt((gt_rotvec - pred_rotvec)**2).mean(dim=[1,2]))
    # This reduction is more accurate because it averages the error per part and then the error across parts
    # It is equivalent to .mean(dim=-1).mean(dim=-1)

    #print(np.size(orientation_error_per_part), orientation_error_per_part)

    #print("orientation_error")
    #print(orientation_error)
    #print()
    #print("orientation_error_new")
    #print(orientation_error_new)
    #print()

    return orientation_error_per_part, orientation_error, orientation_error_new
