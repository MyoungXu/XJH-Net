import torch
import numpy as np


def events_to_image(xs, ys, ps, sensor_size=(720, 1280), interpolation=None, padding=False, meanval=False, default=0):
    """
    Place events into an image using numpy
    @param xs x coords of events
    @param ys y coords of events
    @param ps Event polarities/weights
    @param sensor_size The size of the event camera sensor
    @param interpolation Whether to add the events to the pixels by interpolation (values: None, 'bilinear')
    @param padding If true, pad the output image to include events otherwise warped off sensor
    @param meanval If true, divide the sum of the values by the number of events at that location
    @returns Event image from the input events
    """
    img_size = (sensor_size[0] + 1, sensor_size[1] + 1)
    if interpolation == 'bilinear' and xs.dtype is not torch.long and xs.dtype is not torch.long:
        xt, yt, pt = torch.from_numpy(xs), torch.from_numpy(ys), torch.from_numpy(ps)
        xt, yt, pt = xt.float(), yt.float(), pt.float()
        img = events_to_image_torch(xt, yt, pt, sensor_size=sensor_size, clip_out_of_range=True, interpolation='bilinear', padding=padding)
        img[img == 0] = default
        img = img.numpy()
        if meanval:
            event_count_image = events_to_image_torch(xt, yt, torch.ones_like(xt),
                                                      clip_out_of_range=True, padding=padding)
            event_count_image = event_count_image.numpy()
    else:
        coords = np.stack((ys, xs))
        try:
            abs_coords = np.ravel_multi_index(coords, img_size)
        except ValueError:
            print("Issue with input arrays! minx={}, maxx={}, miny={}, maxy={}, coords.shape={}, \
                    sum(coords)={}, sensor_size={}".format(np.min(xs), np.max(xs), np.min(ys), np.max(ys),
                                                           coords.shape, np.sum(coords), img_size))
            raise ValueError
        img = np.bincount(abs_coords, weights=ps, minlength=img_size[0] * img_size[1])
        img = img.reshape(img_size)
        if meanval:
            event_count_image = np.bincount(abs_coords, weights=np.ones_like(xs), minlength=img_size[0] * img_size[1])
            event_count_image = event_count_image.reshape(img_size)
    if meanval:
        img = np.divide(img, event_count_image, out=np.ones_like(img) * default, where=event_count_image != 0)
    return img[0:sensor_size[0], 0:sensor_size[1]]


def events_to_voxel(xs, ys, ts, ps, B, sensor_size=(720, 1280)):
    """
    Turn set of events to a voxel grid tensor, using temporal bilinear interpolation
    @param xs List of event x coordinates (torch tensor)
    @param ys List of event y coordinates (torch tensor)
    @param ts List of event timestamps (torch tensor)
    @param ps List of event polarities (torch tensor)
    @param B Number of bins in output voxel grids (int)
    @param sensor_size The size of the event sensor/output voxels
    @param temporal_bilinear Whether the events should be naively
        accumulated to the voxels (faster), or properly
        temporally distributed
    @returns Voxel of the events between t0 and t1
    """
    assert (len(xs) == len(ys) and len(ys) == len(ts) and len(ts) == len(ps))
    bins = []
    dt = ts[-1] - ts[0]
    t_norm = (ts - ts[0]) / dt * (B - 1)
    zeros = (np.expand_dims(np.zeros(t_norm.shape[0]), axis=0).transpose()).squeeze()
    for bi in range(B):
        bilinear_weights = np.maximum(zeros, 1.0 - np.abs(t_norm - bi))
        weights = ps * bilinear_weights
        vb = events_to_image(xs.squeeze(), ys.squeeze(), weights.squeeze(), sensor_size=sensor_size, interpolation=None)
        bins.append(vb)
    bins = np.stack(bins)
    return bins


def events_to_image_torch(xs, ys, ps, device=None, sensor_size=(720, 1280), clip_out_of_range=True, interpolation=None, padding=True, default=0):
    """
    Method to turn event tensor to image. Allows for bilinear interpolation.
    @param xs Tensor of x coords of events
    @param ys Tensor of y coords of events
    @param ps Tensor of event polarities/weights
    @param device The device on which the image is. If none, set to events device
    @param sensor_size The size of the image sensor/output image
    @param clip_out_of_range If the events go beyond the desired image size,
       clip the events to fit into the image
    @param interpolation Which interpolation to use. Options=None,'bilinear'
    @param padding If bilinear interpolation, allow padding the image by 1 to allow events to fit:
    @returns Event image from the events
    """
    if device is None:
        device = xs.device
    if interpolation == 'bilinear' and padding:
        img_size = (sensor_size[0] + 1, sensor_size[1] + 1)
    else:
        img_size = list(sensor_size)

    mask = torch.ones(xs.size(), device=device)
    if clip_out_of_range:
        zero_v = torch.tensor([0.], device=device)
        ones_v = torch.tensor([1.], device=device)
        clipx = img_size[1] if interpolation is None and padding == False else img_size[1] - 1
        clipy = img_size[0] if interpolation is None and padding == False else img_size[0] - 1
        mask = torch.where(xs >= clipx, zero_v, ones_v) * torch.where(ys >= clipy, zero_v, ones_v)

    img = (torch.ones(img_size) * default).to(device)
    if interpolation == 'bilinear' and xs.dtype is not torch.long and xs.dtype is not torch.long:
        pxs = (xs.floor()).float()
        pys = (ys.floor()).float()
        dxs = (xs - pxs).float()
        dys = (ys - pys).float()
        pxs = (pxs * mask).long()
        pys = (pys * mask).long()
        masked_ps = ps.squeeze() * mask
        interpolate_to_image(pxs, pys, dxs, dys, masked_ps, img)
    else:
        if xs.dtype is not torch.long:
            xs = xs.long().to(device)
        if ys.dtype is not torch.long:
            ys = ys.long().to(device)
        try:
            mask = mask.long().to(device)
            xs, ys = xs * mask, ys * mask
            img.index_put_((ys, xs), ps, accumulate=True)
        except Exception as e:
            print("Unable to put tensor {} positions ({}, {}) into {}. Range = {},{}".format(
                ps.shape, ys.shape, xs.shape, img.shape, torch.max(ys), torch.max(xs)))
            raise e
    return img


def interpolate_to_image(pxs, pys, dxs, dys, weights, img):
    """
    Accumulate x and y coords to an image using bilinear interpolation
    @param pxs Numpy array of integer typecast x coords of events
    @param pys Numpy array of integer typecast y coords of events
    @param dxs Numpy array of residual difference between x coord and int(x coord)
    @param dys Numpy array of residual difference between y coord and int(y coord)
    @returns Image
    """
    img.index_put_((pys, pxs), weights * (1.0 - dxs) * (1.0 - dys), accumulate=True)
    img.index_put_((pys, pxs + 1), weights * dxs * (1.0 - dys), accumulate=True)
    img.index_put_((pys + 1, pxs), weights * (1.0 - dxs) * dys, accumulate=True)
    img.index_put_((pys + 1, pxs + 1), weights * dxs * dys, accumulate=True)
    return img

