from arc.eval import to_lexicon, to_stream
from arc.io import read_phoneme_corpus, read_syllables_corpus
import click
import logging
import os
import datetime
import json

from arc import load_phonemes, make_syllables, make_words, make_lexicons, make_streams
from arc.types.base_types import Register

ARC_WORKSPACE = "arc_workspace"

logger = logging.getLogger(__name__)


def open_file_in_browser(file_path: str):
    import webbrowser
    webbrowser.open('file://' + os.path.realpath(file_path))


def write_out_streams(streams: Register, save_path: str, open_in_browser: bool = True):
    FILE_NAME = "results.json"

    for i, stream in enumerate(streams):
        with open(os.path.join(save_path, f"stream_{i}.txt"), 'w') as file:
            file.write(stream.id)
        logger.info(f"- {stream.id}")

    with open(os.path.join(save_path, FILE_NAME), 'w') as file:
        results = {"streams": {}, "info": {}}
        for i, stream in enumerate(streams):
            results["streams"][f"stream_{i}"] = {
                "stream": stream.id,
                "lexicon": "|".join([word.id for word in stream.info["lexicon"]]),
                "rhythmicity_indexes": stream.info["rhythmicity_indexes"],
                "stream_tp_mode": stream.info["stream_tp_mode"],
                "n_syllables_per_word": stream.info["n_syllables_per_word"],
                "n_look_back": stream.info["n_look_back"],
                "phonotactic_control": stream.info["phonotactic_control"],
                "syllables_info": stream.info["syllables_info"],
            }
        results["info"] = streams.info
        json.dump(results, file)

    if open_in_browser:
        open_file_in_browser(os.path.join(save_path, FILE_NAME))
        # from filebrowser.app import server
        # server("127.0.0.1", 8080, save_path)


def setup_logging(workspace_path: str):
    logging.basicConfig(filename=os.path.join(workspace_path, "debug.log"), 
                        encoding='utf-8', level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.addHandler(logging.StreamHandler())


def setup_workspace(workspace: str, name="arc_out"):
    workspace_dir = f"{name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    workspace_path = os.path.join(os.path.normpath(workspace), workspace_dir)
    os.makedirs(workspace_path, exist_ok=True)
    return workspace_path


@click.group(help="========== ARC: Artificial languages with Rhythmicity Controls ==========\n\n" 
                  "ARC generates streams of syllables from an artificial language based on a natural language corpus (default: German). "
                  "It will also generate syllables, pseudo-words, and lexicons as byproducts of the artificial language creation.")
def cli():
    pass


@cli.command(help="Evaluate existing lexicon.")
@click.option('--lexicon', type=str, default="lexicon.txt", help="Name of text file containing the lexicon. ARC will look for it in the 'workspace' directory.")
@click.option('--workspace', type=click.Path(), default=ARC_WORKSPACE, help="Write all results and checkpoints.")
@click.option('--ssml/--no-ssml', type=bool, default=True, help="Export syllables to SSML.")
@click.option('--open-browser/--no-open-browser', type=bool, default=True, help="Display results json in default browser.")
def evaluate_lexicon(
    lexicon: str,
    workspace: str,
    ssml: bool,
    open_browser: bool
):
    workspace_path = setup_workspace(workspace, name="arc_eval_lexicon_out")
    setup_logging(workspace_path)

    assert os.path.exists(os.path.join(workspace, lexicon)), f"No file '{lexicon}' found in workspace directory '{os.path.realpath(workspace)}'."

    with open(os.path.join(workspace, lexicon), 'r') as file:
        data = file.read()

    lexicon = to_lexicon([t .split("|") for t in data.split("||")], syllable_type="cv")

    save_path = os.path.join(workspace_path, "lexicon.json")
    logger.info(f"Read Lexicon: {lexicon}")
    lexicon.save(save_path)
    logger.info(f"Lexicon object saved to file: {save_path}")

    syllables = lexicon.flatten()
    syllables.save(os.path.join(workspace_path, "syllables.json"))
    logger.info(f"Syllables object saved to file: {os.path.join(workspace_path, 'syllables.json')}")

    syllables_with_corpus_stats = syllables.intersection(read_syllables_corpus())
    syllables_with_corpus_stats.save(os.path.join(workspace_path, "syllables_with_corpus_stats.json"))
    logger.info(f"Syllables object with corpus stats saved to file: {os.path.join(workspace_path, 'syllables_with_corpus_stats.json')}")

    if ssml:
        from arc.io import export_speech_synthesizer
        export_speech_synthesizer(syllables, syllables_dir=os.path.join(workspace_path, "ssml"))

    phonemes = syllables.flatten()
    phonemes.save(os.path.join(workspace_path, "phonemes.json"))
    logger.info(f"Phonemes object saved to file: {os.path.join(workspace_path, 'phonemes.json')}")

    corpus_phonemes = read_phoneme_corpus()
    phonemes_with_corpus_stats = phonemes.intersection(corpus_phonemes)
    phonemes_with_corpus_stats.save(os.path.join(workspace_path, "phonemes_with_corpus_stats.json"))
    logger.info(f"Phonemes object with corpus stats saved to file: {os.path.join(workspace_path, 'phonemes_with_corpus_stats.json')}")

    if open_browser:
        open_file_in_browser(save_path)


@cli.command(help="Evaluate existing stream.")
@click.option('--stream', type=str, default="stream.txt", help="Name of text file containing the stream. ARC will look for it in the 'workspace' directory.")
@click.option('--workspace', type=click.Path(), default=ARC_WORKSPACE, help="Write all results and checkpoints.")
@click.option('--ssml/--no-ssml', type=bool, default=True, help="Export syllables to SSML.")
@click.option('--open-browser/--no-open-browser', type=bool, default=True, help="Display results json in default browser.")
def evaluate_stream(
    stream: str,
    workspace: str,
    ssml: bool,
    open_browser: bool
):
    workspace_path = setup_workspace(workspace, name="arc_eval_stream_out")
    setup_logging(workspace_path)

    assert os.path.exists(os.path.join(workspace, stream)), f"No file '{stream}' found in workspace directory '{os.path.realpath(workspace)}'."

    with open(os.path.join(workspace, stream), 'r') as file:
        data = file.read()

    stream = to_stream(data.split("|"), syllable_type="cv")

    save_path = os.path.join(workspace_path, "stream.json")
    logger.info(f"Read Stream: {stream}")
    stream.save(save_path)
    logger.info(f"Stream object saved to file: {save_path}")

    syllables = stream.flatten()
    syllables.save(os.path.join(workspace_path, "syllables.json"))
    logger.info(f"Syllables object saved to file: {os.path.join(workspace_path, 'syllables.json')}")

    syllables_with_corpus_stats = syllables.intersection(read_syllables_corpus())
    syllables_with_corpus_stats.save(os.path.join(workspace_path, "syllables_with_corpus_stats.json"))
    logger.info(f"Syllables object with corpus stats saved to file: {os.path.join(workspace_path, 'syllables_with_corpus_stats.json')}")

    if ssml:
        from arc.io import export_speech_synthesizer
        export_speech_synthesizer(syllables, syllables_dir=os.path.join(workspace_path, "ssml"))

    phonemes = syllables.flatten()
    phonemes.save(os.path.join(workspace_path, "phonemes.json"))
    logger.info(f"Phonemes object saved to file: {os.path.join(workspace_path, 'phonemes.json')}")

    corpus_phonemes = read_phoneme_corpus()
    phonemes_with_corpus_stats = phonemes.intersection(corpus_phonemes)
    phonemes_with_corpus_stats.save(os.path.join(workspace_path, "phonemes_with_corpus_stats.json"))
    logger.info(f"Phonemes object with corpus stats saved to file: {os.path.join(workspace_path, 'phonemes_with_corpus_stats.json')}")


    if open_browser:
        open_file_in_browser(save_path)


@cli.command(help="Generate new lexicons and syllable streams.")
@click.option('--workspace', type=click.Path(), default=ARC_WORKSPACE, help="Write all results and checkpoint here.")
@click.option('--ssml/--no-ssml', type=bool, default=True, help="Export syllables to SSML.")
@click.option('--open-browser/--no-open-browser', type=bool, default=True, help="Display results json in default browser.")
def generate(
    workspace: str,
    ssml: bool,
    open_browser: bool,
):
    workspace_path = setup_workspace(workspace, name="arc_generate_out")
    setup_logging(workspace_path)

    phonemes = load_phonemes()
    logger.info(f"Generate Phonemes: {phonemes}")
    phonemes.save(os.path.join(workspace_path, "phonemes.json"))

    syllables = make_syllables(phonemes)
    logger.info(f"Generate Syllables: {syllables}")
    syllables.save(os.path.join(workspace_path, "syllables.json"))

    if ssml:
        from arc.io import export_speech_synthesizer
        export_speech_synthesizer(syllables, syllables_dir=os.path.join(workspace_path, "ssml"))

    logger.info(f"Generate Words: ...")
    words = make_words(syllables)
    logger.info(f"Words: {words}")
    words.save(os.path.join(workspace_path, "words.json"))

    logger.info(f"Generate Lexicons: ...")
    lexicons = make_lexicons(words, n_lexicons=2, n_words=4)
    logger.info(f"Lexicons: {[str(l) for l in lexicons]}")
    for i, lexicon in enumerate(lexicons):
        lexicon.save(os.path.join(workspace_path, f"lexicon_{i}.json"))
    
    logger.info(f"Generate Streams: ...")
    streams = make_streams(lexicons)
    streams.save(os.path.join(workspace_path, f"streams.json"))

    logger.info(f"Streams: ")
    write_out_streams(streams, save_path=workspace_path, open_in_browser=open_browser)
