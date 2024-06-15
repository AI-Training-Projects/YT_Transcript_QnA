# APPLICATION ENHANCEMENTS:

################################################

## CODE, SETUP, AND DOCUMENTATION ENHANCEMENTS:

TODO: Clean up the API calls to be more meaningful
TODO: Add documentation to all functions.
TODO: Make instructions for creating the updated environment and all the packages in it. 
TODO: Test the installation using all CONDA package installations

## FUNCTIONALITY ENHANCEMENTS:
### HIGH PRIORITY: 
TODO: Add caching to save the downloaded YT Audio and converted .wav file

TODO: Save all run data in a "Deliverable Package" in a time-stamped run folder. 
            Input URLs
            (transcript file) QnA prompts and responses 
            Sound and video files
            Application Log files. 

TODO: Try streaming the using the HuggingSound package Speech to Text translation to the Streamlit interface.  

TODO: Ask user if they want to open the YouTube Video in a browser window at the same time 

Add configuration page to specify where to save the package.

### LOW PRIORITY: 
TODO: Make a browser plugin out of this that runs below the video? 

## NEW IDEAS

TODO: Encapsulate all application functionality in your own browser application?
TODO: Annotated Research Run Log system to speak, type, or auto-capture run results summaries in searchable files or transcript database.
TODO: Add a Plugin to the AI Runtime System to Email out the results when finished


LOGGING AND SAVING OUTPUTS: 

Logging is not yet saving to a log file.  Application log files should be saved to a datetime-stamped "run_{date}_num_{run_count}.log" where run_count is the number of times the application has been run that day.

Add code to save all sound input and output files, transcript files, and application logfiles to a date-stamped run_results folder named "results_{date}_run_{run_count}" where run_count is the number of times the application has been run that day.  These folders should be saved in the "run_results_data" folder in the project root folder.  Check to see if these folders already exist at the first run on any given date, and  create them if they do not.

For the transcript name use the following filename string template.
{video_url_plug_name}_YT_Transcript_captured_{Date}.md