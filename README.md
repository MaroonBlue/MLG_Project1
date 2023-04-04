# FastText for Graphs
Project of using FastText for Node Embeddings by Piotr Krzemiński

## Installation Instructions
Simply run the following command inside the project folder:

    $ python -m pip install -r requirements.txt

## GUI / Sentences Generation

Run the following command to open the GUI (assuming rendering is turned on):

    $ python main.py
Following settings are available in the main.py file (FastTextSettings object):

Format of the following list: {settings_name} = {default_value} - {description}
 - render = True - Display the GUI - if disabled, automatically starts sentences generation instead
 - render_isomorphic_graphs = True - Render isomorphisms on the side of the GUI - disabling increases the max rendering fps by reducing the drawing update workload
 - render_auto_walk_delay_seconds = 0.0001 - How often to update the subgraph. Note that this is a delay between each step, not FPS
 - render_x_isomorphisms_per_column = 6 - Display isomorphisms in columns of given amount of elements each, stylistic choice
 - subgraph_size = 5 - Size of the isomorphic subgraphs to generate and walk with
 - letters_per_sentence = 50 - Self explanatory
 - sentences_to_generate = 1000 - Self explanatory
 - spacebar_probability = 0.1 - Chance of inserting spacebar after each symbol
 - end_of_sentence_symbol = '\n' -  Symbol at the end of each sentence
 - file_name = "facebook_dataframe_graph" - Name of the file to use for the output. A '.txt' file extension will be added automatically

Controls of the GUI:

 - X - perform a single random walk without generating a sentence letter
 - C - begin the automatic generation of the sentences. The progress bar with the approximate time can be seen in the console
 - V - stops the automatic generation of the sentences prematurely, saves output to the text file
 - Z - turns off graph refreshing, massively accelerating the generation time

## FastText Classification Task
Simply follow the 'evaluation' Juputer notebook provided in the project files to launch FastText on Facebook Graphs with classification evaluation for friends circles (24 classes)