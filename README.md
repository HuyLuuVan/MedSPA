<div align="center">

<h1 align="center">MedSPA</h1>
<p align="center">Self-planning Medical Agents</p>
<p align="center">
  <img src="assets/hero_logo.png" width="160" alt="MedSPA hero">
</p>

</div>

<div class="section">
  <h2>Abstract</h2>

  <p>
    Reliable medical diagnosis requires a precise final prediction, which is
    strictly supported by observable evidence. Recent large vision-language
    models perform well on medical image understanding tasks, yet they often
    lack explicit control over which evidence to inspect before making a
    prediction. To facilitate flexible planning of evidence exploration, we
    propose MedSPA, a self-planning medical reasoning framework that separates
    diagnostic evidence acquisition from evidence-grounded prediction with two
    cooperative agents: a planning-and-reasoning agent that adaptively decides
    what to inspect and when to stop, and a summary agent that generates the
    final output from the accumulated evidence. MedSPA distills medical
    knowledge from strict diagnostic procedures but acquires flexibility
    through reinforcement learning to generate optimal outputs. Experiments on
    medical report generation and medical visual question answering show that
    MedSPA improves clinical performance, faithfulness, and controllability
    over existing reasoning baselines.
  </p>
</div>