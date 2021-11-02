#TensorFlow is tested and supported on the following 64-bit systems: Python 3.5–3.8 'https://www.tensorflow.org/install' -  30.11.2020
import tensorflow as tf
import tensorflow_hub as hub

from Process_Articles import wikiTrain,wikiEvalTrain,wikiFinalTrain
from Loading_Transcripts import transcripts,transcriptsKeywords
from Loading_Transcripts import dictIndexTranscript,dictTranscriptEmbed

tf.compat.v1.disable_v2_behavior()

# Sometimes data loaded have problems. You have to go in Temp files delete tf_hub folder where this is downloaded.
embed = hub.load("https://tfhub.dev/google/nnlm-es-dim128-with-normalization/2")
X = []
Z = []
Q = []
T = []
K = []


with tf.compat.v1.Session() as session:
    session.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])
    X = session.run(embed(wikiTrain))
    Z = session.run(embed(wikiEvalTrain))
    Q = session.run(embed(wikiFinalTrain))
    T = session.run(embed(transcripts))
    K = session.run(embed(transcriptsKeywords))

counter = 0
for i in range(0, len(T)):
    dictTranscriptEmbed[dictIndexTranscript[counter]] = T[i]
    counter = counter + 1
print(counter)
