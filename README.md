<h1>Membership Inference Attack using LiRA + Confidence Scoring</h1>

<p>
This project shows how machine learning models can unknowingly leak information about their
training data through their prediction practices.
</p>

<p>
We implement a <strong>Membership Inference Attack (MIA)</strong> on a <strong>ResNet-18</strong> model using:
</p>

<ul>
  <li>LiRA (Likelihood Ratio Attack)</li>
  <li>Confidence-based scoring</li>
</ul>

<p>
Our aim is to predict if a given sample was part of the model’s training data.
</p>

<h2>Quick Background</h2>

<p>
Models tend to act differently on data they’ve seen vs. data they haven’t. We use this difference
to determine membership.
</p>

<p>Each sample gets a membership score between 0 and 1:</p>

<ul>
  <li><strong>1</strong> means likely training data</li>
  <li><strong>0</strong> means likely unseen data</li>
</ul>

<h2>Methods</h2>

<ul>
  <li>
    <strong>Confidence Score</strong> — Uses the maximum softmax probability. Higher confidence usually
    means the model has seen the sample during training.
  </li>
  <li>
    <strong>LiRA</strong> — Trained 64 shadow models to simulate how the target model behaves. Then
    compared loss patterns between member and non-member data to decide membership.
  </li>
  <li>
    <strong>Final Score</strong> = 0.5 × LiRA + 0.5 × Confidence. Combining both improves stability and
    accuracy.
  </li>
</ul>

<h2>Setup</h2>

<ul>
  <li>Target model is ResNet-18 for 9-class classification using 32×32 images.</li>
  <li>Trained 64 shadow models, each on 50% random subsets of the dataset.</li>
  <li>Optimizer used is SGD with learning rate = 0.03 and momentum = 0.9.</li>
  <li>Training is run for 50 epochs.</li>
</ul>

<h2>Testing</h2>

<p>
Metric used is <strong>TPR @ 5% FPR = 0.0587 </strong>. This measures how well true training samples are identified
while keeping false positives low.
</p>

<h2>Final Observations</h2>

<ul>
  <li>LiRA performs better than confidence score alone.</li>
  <li>Combining LiRA with confidence gives the best performance.</li>
  <li>Shadow models improve stability and reliability of results.</li>
</ul>

<h2>Structure</h2>

<pre><code>
python train_attack_model.py
submission2.csv</code></pre>

<h2>Takeaway</h2>

<p>
So turns out, machine learning models are not fully private. Even without seeing the training data,
one can still figure out if a sample was part of it, especially when combining LiRA with confidence
scoring.
</p>

<h2>Authors</h2>

<p>
Ananya Bhardwaz — anbh00002@stud.uni-saarland.de<br>
Aryan Aryan — arar00002@stud.uni-saarland.de
</p>
