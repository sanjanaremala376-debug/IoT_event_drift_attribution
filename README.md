# IoT_event_drift_attribution
Drift vs real-event classification and confidence-gated correction on Intel Berkeley Lab sensor data (course project, synthetic injections)
Confidence-Aware Event–Drift Attribution in IoT Sensor Networks

Embedded Systems course project (team of 3). An offline prototype that classifies short windows of temperature-sensor data as normal, sensor drift, a real environmental event, or both, and uses a confidence gate to decide whether to correct a reading, leave it alone, or flag it for review.

Scope: all evaluation uses synthetic drift and event injections on real sensor data. Results are research-prototype evidence, not a validated or deployable system.



THE PROBLEM
Low-cost IoT sensors drift over time. If a real environmental event (for example, a room warming up) is mistaken for drift, "correcting" it removes genuine information. This project classifies short windows of sensor data into four classes and only allows a correction when the evidence is strong.

Class	        Meaning
NORMAL	      no drift, no event
DRIFT	        sensor-specific drift
EVENT	        real environmental change seen by the target and its peer sensors
EVENT_DRIFT	  both at the same time



DATA
Intel Berkeley Research Lab sensor data (54 motes, about 2.3M raw readings, 28 Feb to 3 Apr 2004; this project uses data up to 21 Mar, before the network-wide sensor failure). The raw file (about 150 MB) is not included in this repository. To rerun the notebooks, download data.txt from the link above and place it at data/raw/data.txt.

The raw data has no confirmed fault labels, so drift and event episodes are injected synthetically (18 episodes, each following one target mote and its peer motes).



PIPELINE
Notebook	                            What it does
01 data_exploration	                  Explores the raw sensor data, including the network-wide sensor failure in late March 2004
02 preprocessing	                    Selects and cleans records, builds the 1-minute grid, chronological 70/15/15 split and train-only normalisation
03 scenario_generation_v2	            Builds 9 scenario variants per episode (warming and cooling events, ramp and step drift) from the 18 episodes; checks that                                         earlier variants reproduce exactly
04 windows_features_AF_v2	            30-minute causal windows (5-minute stride) and 19 temporal and peer-comparison features; tests that no future data leaks                                           into features
05 feature_set_comparison_v2	        Random Forest comparison of feature sets A, F, G, H; leave-one-episode-out check across all 18 episodes
06 calibration_confidence_gate_v2	    Probability calibration, confidence gate and a correct / preserve / defer decision rule
07 safe_adaptation_trust_score	      Baselines (fixed threshold, CUSUM, Page-Hinkley), stress tests, correction metrics
08 gateway_edge_profiling_integrated	Inference latency and memory on a desktop (not an embedded device)

The shared helper is in src/intel_utils.py.

Two scripts in scripts/ use only the saved outputs of notebooks 04 and 06, with no retraining and no threshold changes. Run them from the repo root after those notebooks:
d1_eval_compare.py: held-out test evaluation of the gated policy against fixed-threshold, CUSUM and Page-Hinkley baselines. Results go to outputs/deliverable1/.
d2_replay_pipeline.py: replays the test streams window by window through the whole pipeline (model, confidence gate, decision, bounded correction, trust update) and logs the latency. Results go to outputs/deliverable2/.



MAIN FINIDING
A feature that helps on warming events becomes a shortcut that fails on cooling events.

Models were trained only on warming scenarios, then scored on cooling and step scenarios they had never seen (leave-one-episode-out over all 18 episodes). Macro-F1, mean across held-out episodes:

Test condition	                A (original, 16 features)	  F (15 features, with peer_level_rel)	  G (14 features, F without peer_level_rel)
Warming ramp (like training)	  0.594	                      0.624	                                  0.564
Warming step	                  0.651	                      0.681	                                  0.598
Cooling ramp	                  0.351	                      0.350	                                  0.482
Cooling step	                  0.437	                      0.433	                                  0.561

F is the best set on warming but loses 0.27 on cooling ramps. G gives up about 0.06 on warming and loses only 0.08.
The class that matters most is event plus drift. On cooling ramps F recalls 5% of those windows (57% on warming); G recalls 34% (49% on warming).
Compared with set A, G is better on cooling ramps in 18 of 18 held-out episodes (mean macro-F1 gain 0.13) and on cooling steps in 17 of 18 (gain 0.12).

Why: peer_level_rel is the peer level minus the usual level for that hour of day. It is positive for warming events and negative for cooling events (medians +3.16 and -3.47), so a model trained only on warming can use it as a warming cue. G is F with just that feature removed, so the gap points at it, but I did not run a separate test of the mechanism. When the model is trained on all scenario types, F is the better set (0.564 vs 0.498), so this is a generalisation problem rather than a useless feature.



GATED POLICY vs BASELINES (held-out test)

Model trained only on warming scenarios (set F), scored on the four held-out test episodes (2,468 windows covering all four classes, warming and cooling, ramp and step). The table shows how often each method applies a correction. A correction on a NORMAL or pure EVENT window is a false correction. Produced by scripts/d1_eval_compare.py.

Method	                                  Drift corrected	  Normal wrongly corrected	  Pure event wrongly corrected	  Event+drift corrected
Gated policy (proposed)	                  71.8%	            1.5%	                      1.1%	                          47.0%
Same classifier, no gate	                80.4%	            11.4%	                      4.9%	                          54.0%
Fixed threshold	                          81.9%	            5.6%	                      6.9%	                          84.1%
CUSUM (tuned on training data)	          66.4%	            2.8%	                      7.5%	                          65.9%
Page-Hinkley (tuned on training data)	    20.9%	            25.9%	                      26.9%	                          21.3%

The gate trades some drift coverage for far fewer false corrections. 
Against the same classifier without the gate, false corrections fall 7.4x on normal windows and 4.3x on pure events, and drift correction drops from 80.4% to 71.8%.

Against the baselines, the gated policy corrects more drift than CUSUM (71.8% vs 66.4%) with fewer false corrections (1.1% vs 7.5% on pure events). The fixed threshold corrects slightly more drift (81.9%) but wrongly corrects 6.9% of pure events and 5.6% of normal windows.

Event+drift windows contain real drift, so the lower correction rate there (47.0%) is partly restraint and partly missed drift.
The gate defers almost half the windows. It accepts 43% of windows (67% of those correctly classified) and defers 47%. Across all 2,468 windows the classifier's macro-F1 is 0.53, and event+drift recall is only 0.33.

Spread across episodes is large. The proportion of drift windows corrected ranges from 22% to 94% across the four test episodes.

Baselines see one signal (target minus peer median); the classifier sees 15 features. Their parameters were set on training and validation episodes only, to keep false corrections near 5%.

With the notebook 7 settings, CUSUM and Page-Hinkley flag every test window, so they are not a meaningful comparison. The tuned versions above replace them.



END-TO-END REPLAY
scripts/d2_replay_pipeline.py replays the 36 test streams window by window through the saved model, gate, decision rule, bounded correction and trust update, with no ground truth used inside the pipeline.

  Replayed decisions match the notebook 6 decisions on all 2,468 scored windows (100% match, probabilities identical to within 1e-15).
  
  Checks that passed: nothing is corrected when a window is deferred or preserved; every correction stays inside the training-set bound; the raw value is always     kept; trust stays between 0 and 1.
  
  Corrections use a lower threshold (0.49) than the confidence gate (0.62). 433 of the roughly 900 corrections on scored windows happened with confidence below      the gate. This is how the saved decision rule is defined, not an error.
  
  Latency on a Windows 10 machine with 4 logical CPUs: median 70 ms per window on one core (95th percentile 84 ms, 99th percentile 93 ms, 2,628 windows), about      190 MB process memory. Using all cores was slower (median 151 ms on the first 300 windows). This is not gateway hardware.


  
LIMITATIONS
Synthetic injections only; labels are not independently verified.
Only four held-out test episodes; windows overlap, so scores are not statistically independent.
The comparison tables report how often a correction is applied, not how close the corrected value is to the true value.
Gate and baseline results are for the model trained only on warming scenarios.
Latency was measured on a Windows desktop, not on gateway hardware. No edge-deployment claim is made.
The trust score (a per-sensor value from 0 to 1 that falls when drift is detected and recovers on normal readings) is implemented in scripts/d2_replay_pipeline.py with fixed step sizes (0.05 down, 0.01 up) that were not tuned or validated. It shows the mechanism running, not a validated setting.



HOW TO RUN
Python 3.12 with pandas, numpy, scikit-learn, pyarrow, matplotlib, joblib, psutil.
Place data.txt at data/raw/data.txt and keep src/intel_utils.py.
Start Jupyter from the project folder and run the notebooks in numerical order (each saves files used by the next).
