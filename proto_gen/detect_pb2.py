# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: yolo.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


class StereoProjectParams(object):
    def __init__(self, project_dis, project_size, theta_rotate=0):
        self.project_dis = project_dis
        self.project_size = project_size
        self.theta_rotate = theta_rotate

    def __str__(self):
        return "project_dis: %f, project_size: %f, theta_rotate: %f" % (
            self.project_dis, self.project_size, self.theta_rotate)

    def SerializeToString(self):
        return self.__str__()


DESCRIPTOR = _descriptor.FileDescriptor(
    name='yolo.proto',
    package='pano_detection',
    syntax='proto3',
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
    serialized_pb=b'\n\x0c\x64\x65tect.proto\x12\x0epano_detection\"\xae\x01\n\x0eGroundTruthBBX\x12\x0c\n\x04xmin\x18\x01 \x01(\x03\x12\x0c\n\x04ymin\x18\x02 \x01(\x03\x12\x0c\n\x04xmax\x18\x03 \x01(\x03\x12\x0c\n\x04ymax\x18\x04 \x01(\x03\x12\r\n\x05label\x18\x05 \x01(\x03\x12\x10\n\x08x_center\x18\x32 \x01(\x01\x12\x10\n\x08y_center\x18\x33 \x01(\x01\x12\r\n\x05width\x18\x34 \x01(\x01\x12\x0e\n\x06height\x18\x35 \x01(\x01\x12\x12\n\nlabel_name\x18\x36 \x01(\t\"\x90\x01\n\x0c\x44\x61tasetModel\x12\x12\n\nimage_path\x18\x01 \x01(\t\x12=\n\x15ground_truth_bbx_list\x18\x02 \x03(\x0b\x32\x1e.pano_detection.GroundTruthBBX\x12\x16\n\x0eimage_filename\x18\x32 \x01(\t\x12\x15\n\rimage_ndarray\x18\x33 \x01(\x0c\"w\n\x10YoloModelRequest\x12\x12\n\nimage_path\x18\x01 \x01(\t\x12\x12\n\nimage_size\x18\x02 \x01(\x03\x12\x14\n\x0cweights_path\x18\x04 \x01(\t\x12\x12\n\nconf_thres\x18\x05 \x01(\x01\x12\x11\n\tiou_thres\x18\x06 \x01(\x01\"f\n\x0f\x44\x65tectResultBBX\x12\x0c\n\x04xmin\x18\x01 \x01(\x03\x12\x0c\n\x04ymin\x18\x02 \x01(\x03\x12\x0c\n\x04xmax\x18\x03 \x01(\x03\x12\x0c\n\x04ymax\x18\x04 \x01(\x03\x12\r\n\x05label\x18\x05 \x01(\x03\x12\x0c\n\x04\x63onf\x18\x06 \x01(\x01\"h\n\x11YoloModelResponse\x12\x12\n\nimage_path\x18\x01 \x01(\t\x12?\n\x16\x64\x65tect_result_bbx_list\x18\x02 \x03(\x0b\x32\x1f.pano_detection.DetectResultBBX\"\xa2\x01\n\x0eProjectRequest\x12\x38\n\x12pano_dataset_model\x18\x01 \x01(\x0b\x32\x1c.pano_detection.DatasetModel\x12\x13\n\x0bpano_height\x18\x02 \x01(\x03\x12\x12\n\npano_width\x18\x03 \x01(\x03\x12\x16\n\x0eproject_height\x18\x04 \x01(\x03\x12\x15\n\rproject_width\x18\x05 \x01(\x03\"P\n\x0fProjectResponse\x12=\n\x17proj_dataset_model_list\x18\x01 \x03(\x0b\x32\x1c.pano_detection.DatasetModel\"N\n\x18PerspectiveProjectParams\x12\t\n\x01x\x18\x01 \x01(\x01\x12\t\n\x01y\x18\x02 \x01(\x01\x12\t\n\x01z\x18\x03 \x01(\x01\x12\x11\n\tproject_r\x18\x34 \x01(\x01\"\x9b\x01\n\x19PerspectiveProjectRequest\x12\x37\n\x0fproject_request\x18\x01 \x01(\x0b\x32\x1e.pano_detection.ProjectRequest\x12\x45\n\x13project_params_list\x18\x02 \x03(\x0b\x32(.pano_detection.PerspectiveProjectParams\"V\n\x13StereoProjectParams\x12\x13\n\x0bproject_dis\x18\x01 \x01(\x01\x12\x14\n\x0cproject_size\x18\x02 \x01(\x01\x12\x14\n\x0ctheta_rotate\x18\x03 \x01(\x01\"\x91\x01\n\x14StereoProjectRequest\x12\x37\n\x0fproject_request\x18\x01 \x01(\x0b\x32\x1e.pano_detection.ProjectRequest\x12@\n\x13project_params_list\x18\x02 \x03(\x0b\x32#.pano_detection.StereoProjectParams*L\n\x0bProjectType\x12\x0b\n\x07Unknown\x10\x00\x12\x0f\n\x0bPerspective\x10\x01\x12\x0c\n\x08Mercator\x10\x02\x12\x11\n\rStereographic\x10\x03\x62\x06proto3'
)

_PROJECTTYPE = _descriptor.EnumDescriptor(
    name='ProjectType',
    full_name='pano_detection.ProjectType',
    filename=None,
    file=DESCRIPTOR,
    create_key=_descriptor._internal_create_key,
    values=[
        _descriptor.EnumValueDescriptor(
            name='Unknown', index=0, number=0,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key),
        _descriptor.EnumValueDescriptor(
            name='Perspective', index=1, number=1,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key),
        _descriptor.EnumValueDescriptor(
            name='Mercator', index=2, number=2,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key),
        _descriptor.EnumValueDescriptor(
            name='Stereographic', index=3, number=3,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key),
    ],
    containing_type=None,
    serialized_options=None,
    serialized_start=1408,
    serialized_end=1484,
)
_sym_db.RegisterEnumDescriptor(_PROJECTTYPE)

ProjectType = enum_type_wrapper.EnumTypeWrapper(_PROJECTTYPE)
Unknown = 0
Perspective = 1
Mercator = 2
Stereographic = 3

_GROUNDTRUTHBBX = _descriptor.Descriptor(
    name='GroundTruthBBX',
    full_name='pano_detection.GroundTruthBBX',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name='xmin', full_name='pano_detection.GroundTruthBBX.xmin', index=0,
            number=1, type=3, cpp_type=2, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='ymin', full_name='pano_detection.GroundTruthBBX.ymin', index=1,
            number=2, type=3, cpp_type=2, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='xmax', full_name='pano_detection.GroundTruthBBX.xmax', index=2,
            number=3, type=3, cpp_type=2, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='ymax', full_name='pano_detection.GroundTruthBBX.ymax', index=3,
            number=4, type=3, cpp_type=2, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='label', full_name='pano_detection.GroundTruthBBX.label', index=4,
            number=5, type=3, cpp_type=2, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='x_center', full_name='pano_detection.GroundTruthBBX.x_center', index=5,
            number=50, type=1, cpp_type=5, label=1,
            has_default_value=False, default_value=float(0),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='y_center', full_name='pano_detection.GroundTruthBBX.y_center', index=6,
            number=51, type=1, cpp_type=5, label=1,
            has_default_value=False, default_value=float(0),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='width', full_name='pano_detection.GroundTruthBBX.width', index=7,
            number=52, type=1, cpp_type=5, label=1,
            has_default_value=False, default_value=float(0),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='height', full_name='pano_detection.GroundTruthBBX.height', index=8,
            number=53, type=1, cpp_type=5, label=1,
            has_default_value=False, default_value=float(0),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='label_name', full_name='pano_detection.GroundTruthBBX.label_name', index=9,
            number=54, type=9, cpp_type=9, label=1,
            has_default_value=False, default_value=b"".decode('utf-8'),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=33,
    serialized_end=207,
)

_DATASETMODEL = _descriptor.Descriptor(
    name='DatasetModel',
    full_name='pano_detection.DatasetModel',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name='image_path', full_name='pano_detection.DatasetModel.image_path', index=0,
            number=1, type=9, cpp_type=9, label=1,
            has_default_value=False, default_value=b"".decode('utf-8'),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='ground_truth_bbx_list', full_name='pano_detection.DatasetModel.ground_truth_bbx_list', index=1,
            number=2, type=11, cpp_type=10, label=3,
            has_default_value=False, default_value=[],
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='image_filename', full_name='pano_detection.DatasetModel.image_filename', index=2,
            number=50, type=9, cpp_type=9, label=1,
            has_default_value=False, default_value=b"".decode('utf-8'),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='image_ndarray', full_name='pano_detection.DatasetModel.image_ndarray', index=3,
            number=51, type=12, cpp_type=9, label=1,
            has_default_value=False, default_value=b"",
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=210,
    serialized_end=354,
)

_YOLOMODELREQUEST = _descriptor.Descriptor(
    name='YoloModelRequest',
    full_name='pano_detection.YoloModelRequest',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name='image_path', full_name='pano_detection.YoloModelRequest.image_path', index=0,
            number=1, type=9, cpp_type=9, label=1,
            has_default_value=False, default_value=b"".decode('utf-8'),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='image_size', full_name='pano_detection.YoloModelRequest.image_size', index=1,
            number=2, type=3, cpp_type=2, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='weights_path', full_name='pano_detection.YoloModelRequest.weights_path', index=2,
            number=4, type=9, cpp_type=9, label=1,
            has_default_value=False, default_value=b"".decode('utf-8'),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='conf_thres', full_name='pano_detection.YoloModelRequest.conf_thres', index=3,
            number=5, type=1, cpp_type=5, label=1,
            has_default_value=False, default_value=float(0),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='iou_thres', full_name='pano_detection.YoloModelRequest.iou_thres', index=4,
            number=6, type=1, cpp_type=5, label=1,
            has_default_value=False, default_value=float(0),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=356,
    serialized_end=475,
)

_DETECTRESULTBBX = _descriptor.Descriptor(
    name='DetectResultBBX',
    full_name='pano_detection.DetectResultBBX',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name='xmin', full_name='pano_detection.DetectResultBBX.xmin', index=0,
            number=1, type=3, cpp_type=2, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='ymin', full_name='pano_detection.DetectResultBBX.ymin', index=1,
            number=2, type=3, cpp_type=2, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='xmax', full_name='pano_detection.DetectResultBBX.xmax', index=2,
            number=3, type=3, cpp_type=2, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='ymax', full_name='pano_detection.DetectResultBBX.ymax', index=3,
            number=4, type=3, cpp_type=2, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='label', full_name='pano_detection.DetectResultBBX.label', index=4,
            number=5, type=3, cpp_type=2, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='conf', full_name='pano_detection.DetectResultBBX.conf', index=5,
            number=6, type=1, cpp_type=5, label=1,
            has_default_value=False, default_value=float(0),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=477,
    serialized_end=579,
)

_YOLOMODELRESPONSE = _descriptor.Descriptor(
    name='YoloModelResponse',
    full_name='pano_detection.YoloModelResponse',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name='image_path', full_name='pano_detection.YoloModelResponse.image_path', index=0,
            number=1, type=9, cpp_type=9, label=1,
            has_default_value=False, default_value=b"".decode('utf-8'),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='detect_result_bbx_list', full_name='pano_detection.YoloModelResponse.detect_result_bbx_list', index=1,
            number=2, type=11, cpp_type=10, label=3,
            has_default_value=False, default_value=[],
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=581,
    serialized_end=685,
)

_PROJECTREQUEST = _descriptor.Descriptor(
    name='ProjectRequest',
    full_name='pano_detection.ProjectRequest',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name='pano_dataset_model', full_name='pano_detection.ProjectRequest.pano_dataset_model', index=0,
            number=1, type=11, cpp_type=10, label=1,
            has_default_value=False, default_value=None,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='pano_height', full_name='pano_detection.ProjectRequest.pano_height', index=1,
            number=2, type=3, cpp_type=2, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='pano_width', full_name='pano_detection.ProjectRequest.pano_width', index=2,
            number=3, type=3, cpp_type=2, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='project_height', full_name='pano_detection.ProjectRequest.project_height', index=3,
            number=4, type=3, cpp_type=2, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='project_width', full_name='pano_detection.ProjectRequest.project_width', index=4,
            number=5, type=3, cpp_type=2, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=688,
    serialized_end=850,
)

_PROJECTRESPONSE = _descriptor.Descriptor(
    name='ProjectResponse',
    full_name='pano_detection.ProjectResponse',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name='proj_dataset_model_list', full_name='pano_detection.ProjectResponse.proj_dataset_model_list', index=0,
            number=1, type=11, cpp_type=10, label=3,
            has_default_value=False, default_value=[],
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=852,
    serialized_end=932,
)

_PERSPECTIVEPROJECTPARAMS = _descriptor.Descriptor(
    name='PerspectiveProjectParams',
    full_name='pano_detection.PerspectiveProjectParams',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name='x', full_name='pano_detection.PerspectiveProjectParams.x', index=0,
            number=1, type=1, cpp_type=5, label=1,
            has_default_value=False, default_value=float(0),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='y', full_name='pano_detection.PerspectiveProjectParams.y', index=1,
            number=2, type=1, cpp_type=5, label=1,
            has_default_value=False, default_value=float(0),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='z', full_name='pano_detection.PerspectiveProjectParams.z', index=2,
            number=3, type=1, cpp_type=5, label=1,
            has_default_value=False, default_value=float(0),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='project_r', full_name='pano_detection.PerspectiveProjectParams.project_r', index=3,
            number=52, type=1, cpp_type=5, label=1,
            has_default_value=False, default_value=float(0),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=934,
    serialized_end=1012,
)

_PERSPECTIVEPROJECTREQUEST = _descriptor.Descriptor(
    name='PerspectiveProjectRequest',
    full_name='pano_detection.PerspectiveProjectRequest',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name='project_request', full_name='pano_detection.PerspectiveProjectRequest.project_request', index=0,
            number=1, type=11, cpp_type=10, label=1,
            has_default_value=False, default_value=None,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='project_params_list', full_name='pano_detection.PerspectiveProjectRequest.project_params_list',
            index=1,
            number=2, type=11, cpp_type=10, label=3,
            has_default_value=False, default_value=[],
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=1015,
    serialized_end=1170,
)

_STEREOPROJECTPARAMS = _descriptor.Descriptor(
    name='StereoProjectParams',
    full_name='pano_detection.StereoProjectParams',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name='project_dis', full_name='pano_detection.StereoProjectParams.project_dis', index=0,
            number=1, type=1, cpp_type=5, label=1,
            has_default_value=False, default_value=float(0),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='project_size', full_name='pano_detection.StereoProjectParams.project_size', index=1,
            number=2, type=1, cpp_type=5, label=1,
            has_default_value=False, default_value=float(0),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='theta_rotate', full_name='pano_detection.StereoProjectParams.theta_rotate', index=2,
            number=3, type=1, cpp_type=5, label=1,
            has_default_value=False, default_value=float(0),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=1172,
    serialized_end=1258,
)

_STEREOPROJECTREQUEST = _descriptor.Descriptor(
    name='StereoProjectRequest',
    full_name='pano_detection.StereoProjectRequest',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name='project_request', full_name='pano_detection.StereoProjectRequest.project_request', index=0,
            number=1, type=11, cpp_type=10, label=1,
            has_default_value=False, default_value=None,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='project_params_list', full_name='pano_detection.StereoProjectRequest.project_params_list', index=1,
            number=2, type=11, cpp_type=10, label=3,
            has_default_value=False, default_value=[],
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=1261,
    serialized_end=1406,
)

_DATASETMODEL.fields_by_name['ground_truth_bbx_list'].message_type = _GROUNDTRUTHBBX
_YOLOMODELRESPONSE.fields_by_name['detect_result_bbx_list'].message_type = _DETECTRESULTBBX
_PROJECTREQUEST.fields_by_name['pano_dataset_model'].message_type = _DATASETMODEL
_PROJECTRESPONSE.fields_by_name['proj_dataset_model_list'].message_type = _DATASETMODEL
_PERSPECTIVEPROJECTREQUEST.fields_by_name['project_request'].message_type = _PROJECTREQUEST
_PERSPECTIVEPROJECTREQUEST.fields_by_name['project_params_list'].message_type = _PERSPECTIVEPROJECTPARAMS
_STEREOPROJECTREQUEST.fields_by_name['project_request'].message_type = _PROJECTREQUEST
_STEREOPROJECTREQUEST.fields_by_name['project_params_list'].message_type = _STEREOPROJECTPARAMS
DESCRIPTOR.message_types_by_name['GroundTruthBBX'] = _GROUNDTRUTHBBX
DESCRIPTOR.message_types_by_name['DatasetModel'] = _DATASETMODEL
DESCRIPTOR.message_types_by_name['YoloModelRequest'] = _YOLOMODELREQUEST
DESCRIPTOR.message_types_by_name['DetectResultBBX'] = _DETECTRESULTBBX
DESCRIPTOR.message_types_by_name['YoloModelResponse'] = _YOLOMODELRESPONSE
DESCRIPTOR.message_types_by_name['ProjectRequest'] = _PROJECTREQUEST
DESCRIPTOR.message_types_by_name['ProjectResponse'] = _PROJECTRESPONSE
DESCRIPTOR.message_types_by_name['PerspectiveProjectParams'] = _PERSPECTIVEPROJECTPARAMS
DESCRIPTOR.message_types_by_name['PerspectiveProjectRequest'] = _PERSPECTIVEPROJECTREQUEST
DESCRIPTOR.message_types_by_name['StereoProjectParams'] = _STEREOPROJECTPARAMS
DESCRIPTOR.message_types_by_name['StereoProjectRequest'] = _STEREOPROJECTREQUEST
DESCRIPTOR.enum_types_by_name['ProjectType'] = _PROJECTTYPE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

GroundTruthBBX = _reflection.GeneratedProtocolMessageType('GroundTruthBBX', (_message.Message,), {
    'DESCRIPTOR': _GROUNDTRUTHBBX,
    '__module__': 'detect_pb2'
    # @@protoc_insertion_point(class_scope:pano_detection.GroundTruthBBX)
})
_sym_db.RegisterMessage(GroundTruthBBX)

DatasetModel = _reflection.GeneratedProtocolMessageType('DatasetModel', (_message.Message,), {
    'DESCRIPTOR': _DATASETMODEL,
    '__module__': 'detect_pb2'
    # @@protoc_insertion_point(class_scope:pano_detection.DatasetModel)
})
_sym_db.RegisterMessage(DatasetModel)

YoloModelRequest = _reflection.GeneratedProtocolMessageType('YoloModelRequest', (_message.Message,), {
    'DESCRIPTOR': _YOLOMODELREQUEST,
    '__module__': 'detect_pb2'
    # @@protoc_insertion_point(class_scope:pano_detection.YoloModelRequest)
})
_sym_db.RegisterMessage(YoloModelRequest)

DetectResultBBX = _reflection.GeneratedProtocolMessageType('DetectResultBBX', (_message.Message,), {
    'DESCRIPTOR': _DETECTRESULTBBX,
    '__module__': 'detect_pb2'
    # @@protoc_insertion_point(class_scope:pano_detection.DetectResultBBX)
})
_sym_db.RegisterMessage(DetectResultBBX)

YoloModelResponse = _reflection.GeneratedProtocolMessageType('YoloModelResponse', (_message.Message,), {
    'DESCRIPTOR': _YOLOMODELRESPONSE,
    '__module__': 'detect_pb2'
    # @@protoc_insertion_point(class_scope:pano_detection.YoloModelResponse)
})
_sym_db.RegisterMessage(YoloModelResponse)

ProjectRequest = _reflection.GeneratedProtocolMessageType('ProjectRequest', (_message.Message,), {
    'DESCRIPTOR': _PROJECTREQUEST,
    '__module__': 'detect_pb2'
    # @@protoc_insertion_point(class_scope:pano_detection.ProjectRequest)
})
_sym_db.RegisterMessage(ProjectRequest)

ProjectResponse = _reflection.GeneratedProtocolMessageType('ProjectResponse', (_message.Message,), {
    'DESCRIPTOR': _PROJECTRESPONSE,
    '__module__': 'detect_pb2'
    # @@protoc_insertion_point(class_scope:pano_detection.ProjectResponse)
})
_sym_db.RegisterMessage(ProjectResponse)

PerspectiveProjectParams = _reflection.GeneratedProtocolMessageType('PerspectiveProjectParams', (_message.Message,), {
    'DESCRIPTOR': _PERSPECTIVEPROJECTPARAMS,
    '__module__': 'detect_pb2'
    # @@protoc_insertion_point(class_scope:pano_detection.PerspectiveProjectParams)
})
_sym_db.RegisterMessage(PerspectiveProjectParams)

PerspectiveProjectRequest = _reflection.GeneratedProtocolMessageType('PerspectiveProjectRequest', (_message.Message,), {
    'DESCRIPTOR': _PERSPECTIVEPROJECTREQUEST,
    '__module__': 'detect_pb2'
    # @@protoc_insertion_point(class_scope:pano_detection.PerspectiveProjectRequest)
})
_sym_db.RegisterMessage(PerspectiveProjectRequest)

# StereoProjectParams = _reflection.GeneratedProtocolMessageType('StereoProjectParams', (_message.Message,), {
#     'DESCRIPTOR': _STEREOPROJECTPARAMS,
#     '__module__': 'detect_pb2'
#     # @@protoc_insertion_point(class_scope:pano_detection.StereoProjectParams)
# })
# _sym_db.RegisterMessage(StereoProjectParams)

StereoProjectRequest = _reflection.GeneratedProtocolMessageType('StereoProjectRequest', (_message.Message,), {
    'DESCRIPTOR': _STEREOPROJECTREQUEST,
    '__module__': 'detect_pb2'
    # @@protoc_insertion_point(class_scope:pano_detection.StereoProjectRequest)
})
_sym_db.RegisterMessage(StereoProjectRequest)

# @@protoc_insertion_point(module_scope)
