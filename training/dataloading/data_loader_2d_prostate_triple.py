import numpy as np
import torch
from threadpoolctl import threadpool_limits

# from nnunetv2.training.dataloading.base_data_loader_prostate import nnUNetDataLoaderBase
from nnunetv2.training.dataloading.base_data_loader_prostate_triple import nnUNetDataLoaderBase
from nnunetv2.training.dataloading.nnunet_prostate_triple import nnUNetDataset_Prostate_Triple


class nnUNetDataLoader2D(nnUNetDataLoaderBase):
    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        seg_all2 = np.zeros(self.seg_shape, dtype=np.int16)
        seg_all3 = np.zeros(self.seg_shape, dtype=np.int16)
        case_properties = []

        for j, current_key in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = self.get_do_oversample(j)
            data, seg, seg2, seg3, properties = self._data.load_case(current_key)
            case_properties.append(properties)

            # select a class/region first, then a slice where this class is present, then crop to that area
            if not force_fg:
                if self.has_ignore:
                    selected_class_or_region = self.annotated_classes_key if (
                            len(properties['class_locations'][self.annotated_classes_key]) > 0) else None
                else:
                    selected_class_or_region = None
            else:
                # filter out all classes that are not present here
                eligible_classes_or_regions = [i for i in properties['class_locations'].keys() if len(properties['class_locations'][i]) > 0]

                # if we have annotated_classes_key locations and other classes are present, remove the annotated_classes_key from the list
                # strange formulation needed to circumvent
                # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
                tmp = [i == self.annotated_classes_key if isinstance(i, tuple) else False for i in eligible_classes_or_regions]
                if any(tmp):
                    if len(eligible_classes_or_regions) > 1:
                        eligible_classes_or_regions.pop(np.where(tmp)[0][0])

                selected_class_or_region = eligible_classes_or_regions[np.random.choice(len(eligible_classes_or_regions))] if \
                    len(eligible_classes_or_regions) > 0 else None

            if selected_class_or_region is not None:
                selected_slice = np.random.choice(properties['class_locations'][selected_class_or_region][:, 1])
            else:
                selected_slice = np.random.choice(len(data[0]))

            data = data[:, selected_slice]
            seg = seg[:, selected_slice]
            seg2 = seg2[:, selected_slice]
            seg3 = seg3[:, selected_slice]

            # the line of death lol
            # this needs to be a separate variable because we could otherwise permanently overwrite
            # properties['class_locations']
            # selected_class_or_region is:
            # - None if we do not have an ignore label and force_fg is False OR if force_fg is True but there is no foreground in the image
            # - A tuple of all (non-ignore) labels if there is an ignore label and force_fg is False
            # - a class or region if force_fg is True
            class_locations = {
                selected_class_or_region: properties['class_locations'][selected_class_or_region][properties['class_locations'][selected_class_or_region][:, 1] == selected_slice][:, (0, 2, 3)]
            } if (selected_class_or_region is not None) else None

            # print(properties)
            shape = data.shape[1:]
            dim = len(shape)
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg if selected_class_or_region is not None else False,
                                               class_locations, overwrite_class=selected_class_or_region)


            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
            valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)
            this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            data = data[this_slice]

            this_slice = tuple([slice(0, seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            seg = seg[this_slice]
            seg2 = seg2[this_slice]
            seg3 = seg3[this_slice]

            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            data_all[j] = np.pad(data, ((0, 0), *padding), 'constant', constant_values=0)
            seg_all[j] = np.pad(seg, ((0, 0), *padding), 'constant', constant_values=-1)
            seg_all2[j] = np.pad(seg2, ((0, 0), *padding), 'constant', constant_values=-1)
            seg_all3[j] = np.pad(seg3, ((0, 0), *padding), 'constant', constant_values=-1)

        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):

                    data_all = torch.from_numpy(data_all).float()
                    seg_all = torch.from_numpy(seg_all).to(torch.int16)
                    seg_all2 = torch.from_numpy(seg_all2).to(torch.int16)
                    seg_all3 = torch.from_numpy(seg_all3).to(torch.int16)
                    images = []
                    segs = []
                    segs2 = []
                    segs3 = []
                    for b in range(self.batch_size):
                        # print(f"data_loader_2d_prostate: {seg_all[b].shape}, {seg_all2[b].shape}")
                        temp_merge = torch.cat((seg_all[b], seg_all2[b], seg_all3[b]), dim=0)
                        # print(f"data_loader_2d_prostate.merged: {temp_merge.shape}")
                        # tmp = self.transforms(**{'image': data_all[b], 'segmentation': seg_all[b]})
                        tmp = self.transforms(**{'image': data_all[b], 'segmentation': temp_merge})
                        # print(f"data_loader_2d_prostate.unmerged: {type(tmp['segmentation'])}, {len(tmp['segmentation']), tmp['segmentation'][0].shape, tmp['segmentation'][1].shape}")
                        # <class 'list'>, (4, torch.Size([2, 64, 80]), torch.Size([2, 32, 40]))
                        images.append(tmp['image'])
                        if isinstance(tmp['segmentation'], list):
                            temp_seg = [item[0].unsqueeze(0) for item in tmp['segmentation']]
                            temp_seg2 = [item[1].unsqueeze(0) for item in tmp['segmentation']]
                            temp_seg3 = [item[2].unsqueeze(0) for item in tmp['segmentation']]
                            # print(f"temp_seg: {len(temp_seg), temp_seg[0].shape, temp_seg[1].shape}")
                        else:
                            temp_seg = tmp['segmentation'][0].unsqueeze(0)
                            temp_seg2 = tmp['segmentation'][1].unsqueeze(0)
                            temp_seg3 = tmp['segmentation'][2].unsqueeze(0)
                        segs.append(temp_seg)
                        segs2.append(temp_seg2)
                        segs3.append(temp_seg3)

                    data_all = torch.stack(images)
                    if isinstance(segs[0], list):
                        seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                        seg_all2 = [torch.stack([s[i] for s in segs2]) for i in range(len(segs2[0]))]
                        seg_all3 = [torch.stack([s[i] for s in segs3]) for i in range(len(segs3[0]))]
                    else:
                        seg_all = torch.stack(segs)
                        seg_all2 = torch.stack(segs2)
                        seg_all3 = torch.stack(segs3)
                    del segs, images, segs2, seg3

            return {'data': data_all, 'target': seg_all, 'target2': seg_all2, 'target3': seg_all3, 'keys': selected_keys}

        return {'data': data_all, 'target': seg_all, 'target2': seg_all2, 'target3': seg_all3, 'keys': selected_keys}


if __name__ == '__main__':
    folder = '/home/litaozhao/nnU-Net/nnUNet_v2/nnunetv2/nnUNetFrame/DATASET/nnUNet_preprocessed/Dataset510_dual_decoder/nnUNetPlans_2d'
    ds = nnUNetDataset_Prostate_Triple(folder, None, 1000)  # this should not load the properties!
    dl = nnUNetDataLoader2D(ds, 366, (65, 65), (56, 40), None, None, None)
    a = next(dl)
