Rainfall Model API
==================

.. currentmodule:: rwgen

.. autoclass:: RainfallModel
   :members: reference_statistics, parameters, fitted_statistics,
             simulated_statistics, parameter_names, unique_seasons
   
   .. automethod:: __init__
   
   .. rubric:: Methods

   .. autosummary::
      :toctree: generated

      ~RainfallModel.__init__
      ~RainfallModel.preprocess
      ~RainfallModel.fit
      ~RainfallModel.simulate
      ~RainfallModel.postprocess
      ~RainfallModel.set_parameters
      ~RainfallModel.set_statistics
      ~RainfallModel.perturb_statistics
      ~RainfallModel.plot
      ~RainfallModel.update_output_folder
      ~RainfallModel.update_simulation_config
      ~RainfallModel.update_shuffling_config
   
   .. rubric:: Attributes

   .. autosummary::

      ~RainfallModel.reference_statistics
      ~RainfallModel.parameters
      ~RainfallModel.fitted_statistics
      ~RainfallModel.simulated_statistics
      ~RainfallModel.parameter_names
      ~RainfallModel.unique_seasons
