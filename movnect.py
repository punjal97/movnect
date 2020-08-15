import keras.backend as K
from keras.applications import MobileNetV2


class MoVnect(object):
    """docstring for MoVnect"""

    def __init__(self, input_shape=(368, 368), number_of_joints=21):
        super(MoVnect, self).__init__()
        self.number_of_joints = number_of_joints
        self.input_shape = input_shape
        self.eight_of_input_size = int(input_shape[0]/8)

    def create_network(self):

        img_size = (input_shape[1], input_shape[0])
        input_shape = (input_shape[1], input_shape[0], 3)
        mobilenetv2_input_shape = (224, 224, 3)

        Input0 = Input(input_shape)
        mobilenetv2 = MobileNetV2(
            input_shape=mobilenetv2_input_shape, include_top=False, weights='imagenet')
        FeatureExtractor = Model(
            inputs=mobilenetv2.input, outputs=mobilenetv2.get_layer('block_12_add').output)

        x = FeatureExtractor(Input0)

        some_layer = layers.Conv2D(368,
                                   kernel_size=(1, 1),
                                   kernel_initializer='he_normal',
                                   name='block13_1a_a')(x)

        some_layer = layers.Activation('relu')(some_layer)

        some_layer = layers.DepthwiseConv2D(
            368, kernel_size=(3, 3), name='block13_1a_b')(some_layer)

        some_layer = layers.Activation('relu')(some_layer)

        some_layer = layers.Conv2D(256, kernel_size=(
            1, 1), kernel_initializer='he_normal', name='block13_1a_c')(some_layer)

        some_layer2 = layers.Conv2D(256, kernel_size=(
            1, 1), kernel_initializer='he_normal', name='block13_2a')(x)

        block13_a = layers.add([some_layer, some_layer2], name='block13a_add')
        block13_a = layers.Activation('relu')(block13_a)

        block13_b = layers.Conv2D(192, kernel_size=(
            1, 1), kernel_initializer='he_normal', name='block13_b_a')(block13_a)

        block13_b = layers.Activation('relu')(block13_b)

        block13_b = layers.DepthwiseConv2D(
            192, kernel_size=(3, 3), name='block13_b_b')(block13_b)

        block13_b = layers.Activation('relu')(block13_b)

        block13_b = layers.Conv2D(192, kernel_size=(
            1, 1), name='block13_b_c')(block13_b)

        block13_b = layers.Activation('relu')(block13_b)

        block13_c1 = layers.Conv2DTranspose(128, kernel_size=(3, 3), strides=(
            2, 2), padding='same', kernel_initializer='he_normal', name='block13_c1', use_bias=False)(block13_b)

        block13_c1 = layers.BatchNormalization(
            fused=True, name='bnblock13_c1')(block13_c1)

        block13_c1 = layers.Activation('relu')(block13_c1)

        block13_c2 = layers.Conv2DTranspose(self.number_of_joints*3, kernel_size=(3, 3), strides=(
            2, 2), padding='same', kernel_initializer='he_normal', name='block13_c2')

        block13_c2_sqr = layers.Lambda(self.square_tensor, output_shape=(self.eight_of_input_size, self.eight_of_input_size,
                                                                         self.number_of_joints * 3))(block13_c2)

        delta_x_sqr = layers.Lambda(
            self.slice_tensor,
            arguments={'k': int(0)},
            output_shape=(self.eight_of_input_size, self.eight_of_input_size, self.number_of_joints))(block13_c2_sqr)

        delta_y_sqr = layers.Lambda(
            self.slice_tensor,
            arguments={'k': int(self.number_of_joints)},
            output_shape=(self.eight_of_input_size, self.eight_of_input_size,
                          self.number_of_joints))(block13_c2_sqr)

        delta_z_sqr = layers.Lambda(
            self.slice_tensor,
            arguments={'k': int(self.number_of_joints*2)},
            output_shape=(self.eight_of_input_size, self.eight_of_input_size,
                          self.number_of_joints))(block13_c2_sqr)

        bone_length_sqr = layers.Add()([delta_x_sqr, delta_y_sqr, delta_z_sqr])
        bone_length = layers.Lambda(
            self.sqrt_tensor,
            output_shape=(self.eight_of_input_size, self.eight_of_input_size, self.number_of_joints))(bone_length_sqr)

        block13_c3 = layers.concatenate([block13_c1, block13_c2, bone_length])

        block14 = layers.Conv2D(128, kernel_size=(
            1, 1), padding='same', kernel_initializaer='he_normal', name='block14_a')(block13_c3)

        block14 = layers.Activation('relu')(block14)

        block14 = layers.DepthwiseConv2D(
            128, kernel_size=(3, 3), name='block14_b')(block14)

        block14 = layers.Activation('relu')(block14)

        featuremaps = layers.Conv2D(self.number_of_joints * 4,
                                    kernel_size=(1, 1),
                                    kernel_initializer='he_normal',
                                    name='res5c_branch2c',
                                    use_bias=False)(block14)

        heatmap_2d = layers.Lambda(self.slice_tensor,
                                   arguments={'k': int(0)},
                                   output_shape=(self.eight_of_input_size,
                                                 self.eight_of_input_size,
                                                 self.number_of_joints),
                                   name='heatmap')(featuremaps)
        loc_heatmap_x = layers.Lambda(self.slice_tensor,
                                      arguments={
                                          'k': int(self.number_of_joints)},
                                      output_shape=(self.eight_of_input_size,
                                                    self.eight_of_input_size,
                                                    self.number_of_joints),
                                      name='x_heatmap')(featuremaps)
        loc_heatmap_y = layers.Lambda(self.slice_tensor,
                                      arguments={
                                          'k': int(self.number_of_joints * 2)},
                                      output_shape=(self.eight_of_input_size,
                                                    self.eight_of_input_size,
                                                    self.number_of_joints),
                                      name='y_heatmap')(featuremaps)
        loc_heatmap_z = layers.Lambda(self.slice_tensor,
                                      arguments={
                                          'k': int(self.number_of_joints * 3)},
                                      output_shape=(self.eight_of_input_size,
                                                    self.eight_of_input_size,
                                                    self.number_of_joints),
                                      name='z_heatmap')(featuremaps)

        m = models.Model(inputs=Input0, outputs=[
                         heatmap_2d, loc_heatmap_x, loc_heatmap_y, loc_heatmap_z])
        m.summary()

        return m
