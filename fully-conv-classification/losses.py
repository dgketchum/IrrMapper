import keras.backend as K
import tensorflow as tf
from sklearn.metrics import confusion_matrix

_epsilon = tf.convert_to_tensor(K.epsilon(), tf.float32)

def binary_focal_loss(gamma=2, alpha=0.25):
    ''' 
    Focal loss:

       FL (pt) = -(1-pt)^gamma * log(pt)
       where
       pt = p if y==1
            1-p otherwise
    '''

    def bfl(y_true, y_pred):
        mask = tf.not_equal(y_true, -1) # true where the mask isn't==-1
        y_pred = tf.nn.sigmoid(y_pred)
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)
        pt_1 = tf.boolean_mask(pt_1, mask)
        pt_0 = tf.boolean_mask(pt_0, mask)

        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               -K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return bfl


def precision_and_recall(y_true, y_pred):
    y_true_sum = tf.math.reduce_sum(y_true, axis=-1)
    mask = tf.not_equal(y_true_sum, 0)
    y_pred = tf.nn.softmax(y_pred, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.boolean_mask(y_pred, mask)
    y_true = tf.boolean_mask(y_true, mask)
    # precision: out of everything I predicted irrigated, what was actually irrigated?
    return 1



def multiclass_focal_loss(gamma=2, alpha=0.25):


    def multiclass_FL(y_true, y_pred):
        y_true_sum = tf.math.reduce_sum(y_true, axis=-1)
        mask = tf.not_equal(y_true_sum, 0)
        # y_true = tf.boolean_mask(y_true, mask)
        # y_pred = tf.boolean_mask(y_pred, mask)
        # probabilities = tf.nn.softmax(y_pred)
        # xen = -y_true * tf.math.log(probabilities) # all 0s where y_true is all 0s
        # loss = alpha*tf.math.pow(1-probabilities, gamma) * xen 
        # return tf.math.reduce_mean(loss)
        probabilities = tf.nn.softmax(y_pred, axis=-1)
        xen = -y_true * tf.math.log(probabilities) # all 0s where y_true is all 0s
        complement = tf.dtypes.cast(tf.equal(y_true, 0), tf.float32)
        negative_probabilities = -tf.math.pow(complement*probabilities,
                gamma)*tf.math.log(complement)
        masked_xen = tf.boolean_mask(xen, mask)
        masked_complement = tf.boolean_mask(xen, negative_probabilities)
        return tf.reduce_mean(masked_xen) + tf.reduce_mean(masked_complement)

    return multiclass_FL


def dice_coef(y_true, y_pred, smooth=1):
        """
        Dice = (2*|X & Y|)/ (|X|+ |Y|)
             =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
        ref: https://arxiv.org/pdf/1606.04797v1.pdf
        """
        intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
        return (2. * intersection + smooth) / (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)


def dice_loss(y_true, y_pred):
    mask = tf.not_equal(y_true, -1)
    # y_pred = tf.nn.softmax(y_pred)
    y_true = tf.boolean_mask(y_true, mask)
    y_pred = tf.boolean_mask(y_pred, mask)
    return 1 - dice_coef(y_true, y_pred)


def masked_binary_xent(pos_weight=1.0):
    # One_hot matrix is all zeros along depth if there isn't
    # a data pixel there. Accordingly, we 
    # mask out the pixels that do not contain data.
    # binary xent requires a y_true of shape nxmx1, with -1
    # indicating nodata
    def mb(y_true, y_pred):
        mask = tf.not_equal(y_true, -1)
        y_true = tf.boolean_mask(y_true, mask)
        y_pred = tf.boolean_mask(y_pred, mask)
        return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
                labels=y_true, 
                logits=y_pred, 
                pos_weight=pos_weight))
    return mb



def masked_categorical_xent(y_true, y_pred):
    # One_hot matrix is all zeros along depth if there isn't
    # a data pixel there. Accordingly, we 
    # mask out the pixels that do not contain data.
    # wait what? I don't need to even mask this!
    # the one_hot matrix contains depthwise 0s 
    # where there isn't data...
    y_true_sum = tf.math.reduce_sum(y_true, axis=-1)
    mask = tf.not_equal(y_true_sum, 0)
    y_true = tf.boolean_mask(y_true, mask)
    y_pred = tf.boolean_mask(y_pred, mask)
    return tf.nn.softmax_cross_entropy_with_logits_v2(y_true, y_pred)


def binary_acc(y_true, y_pred):
    y_pred = tf.round(tf.nn.sigmoid(y_pred))
    mask = tf.not_equal(y_true, -1)
    y_true = tf.boolean_mask(y_true, mask)
    y_pred = tf.boolean_mask(y_pred, mask)
    return K.mean(K.equal(y_true, K.round(y_pred)))


def m_acc(y_true, y_pred):
    y_true_sum = tf.reduce_sum(y_true, axis=-1)
    mask = tf.not_equal(y_true_sum, 0)
    y_pred = tf.nn.softmax(y_pred)
    y_pred = tf.argmax(y_pred, axis=-1)
    y_true = tf.argmax(y_true, axis=-1)
    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)
    return K.mean(K.equal(y_pred_masked, y_true_masked))
