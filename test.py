import tensorflow as tf
with tf.Graph().as_default():
	matrix=tf.constant([[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16]],dtype=tf.int32)
	reshaped_2x8_matrix=tf.reshape(matrix,[2,8])
	reshaped_4x4_matrix=tf.reshape(matrix,[4,4])
	reshaped_2x2x4_matrix=tf.reshape(matrix,[2,2,4])
	one_dimensional_vector=tf.reshape(matrix,[16])
	with tf.Session() as sess:
		print('Original matrix (8x2)')
		print(matrix.eval())
		print('Reshaped matrix (2x8)')
		print(reshaped_2x8_matrix.eval())
		print('Reshaped matrix (4x4)')
		print(reshaped_4x4_matrix.eval())
		print(reshaped_2x2x4_matrix)
  
