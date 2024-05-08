"""Observe that this file can't contain local functions to avoid circular imports"""

import io
import os
from io import BytesIO
import shutil
import logging
from azure.storage.blob import BlobServiceClient

from PyPDF2 import PdfReader, PdfWriter
from docx2pdf import convert
from langchain_community.embeddings import AzureOpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import PGVector, FAISS
from langchain_text_splitters import CharacterTextSplitter
from typing import List, Tuple, Any

from conf import configs as conf_


# TODO: REFACTOR THIS

class DataReader:
    def __init__(self, configs):
        blob_service_client = BlobServiceClient.from_connection_string(configs.blob_connection_string)
        container_client = blob_service_client.get_container_client(configs.blob_container_name)
        self.container_client = container_client

    @staticmethod
    def extract_text_from_pdf(blob_data) -> str:
        """
        Extracts text from a PDF blob data.

        Args:
            blob_data: The blob data of the PDF.

        Returns:
            str: The extracted text from the PDF.
        """
        raw_text = ""
        with io.BytesIO() as f:
            blob_data.readinto(f)
            pdf_reader = PdfReader(f)
            for page in pdf_reader.pages:
                raw_text += page.extract_text()
        return raw_text

    @staticmethod
    def read_text_and_metadata_from_pdf_with_meta(file_path: str) -> List[Tuple[str, dict]]:
        """
        Extracts text and metadata from a PDF file.

        Args:
            file_path (str): Path to the PDF file.

        Returns:
            List[Tuple[str, dict]]: A list of tuples containing text and metadata for each page of the PDF.
        """
        pages_data = []

        with open(file_path, 'rb') as f:
            pdf_reader = PdfReader(f)

            for page_number, page in enumerate(pdf_reader.pages, start=1):
                page_text = page.extract_text()
                main_title = pdf_reader.metadata.get('/Title', 'Untitled')

                page_metadata = {
                    "title": main_title,
                    "page": page_number
                }

                pages_data.append((page_text, page_metadata))

        return pages_data

    @staticmethod
    def read_text_and_metadata_from_pdf(blob_data) -> List[Tuple[str, dict]]:
        """
        Extracts text and metadata (such as page number and title) from a PDF blob data.

        Args:
            blob_data: The blob data of the PDF.

        Returns:
            List[Tuple[str, dict]]: A list of tuples containing the extracted text and metadata for each page.
        """

        pages_data = []
        with io.BytesIO() as f:
            blob_data.readinto(f)
            pdf_reader = PdfReader(f)

            for page_number, page in enumerate(pdf_reader.pages, start=1):
                page_text = page.extract_text()

                # Extract the first title from each page as main title
                page_lines = page_text.split("\n")
                if page_number == 1:
                    try:
                        cleaned_page_lines = [line.strip() for line in page_lines if line.strip()]

                        # main_title = cleaned_page_lines[1]
                        main_title = '-'.join(cleaned_page_lines[1:3])

                    except IndexError:
                        main_title = 'Untitled'

                page_metadata = {
                    "title": main_title if main_title else "Untitled",
                    "page": page_number
                }
                pages_data.append((page_text, page_metadata))

        return pages_data

    # TODO:REMOVE/REFACTOR THIS FUNC
    def read_text_from_blob_storage(self) -> str:
        """
        Reads text data from PDF files stored in Azure Blob Storage.

        Returns:
            str: The concatenated text extracted from the PDF files.
        """

        raw_text = ""
        blob_list = self.container_client.list_blobs()
        for blob in blob_list:
            if blob.name.endswith(".pdf"):
                blob_client = self.container_client.get_blob_client(blob)
                blob_data = blob_client.download_blob()
                raw_text += self.extract_text_from_pdf(blob_data)

        return raw_text

    def read_text_and_metadata(self, suffix=None) -> List[Tuple[str, dict]]:
        """
        Reads text and metadata from PDF files stored in Azure Blob Storage.

        Returns:
            List[Tuple[str, dict]]: A list of tuples containing the extracted text and metadata for each page.
        """

        pages_data = []

        if suffix is not None:
            for file_name in os.listdir(suffix):
                full_path = os.path.join(suffix, file_name)
                pages_data.extend(self.read_text_and_metadata_from_pdf_with_meta(full_path))
        else:
            blob_list = self.container_client.list_blobs()
            for blob in blob_list:
                if blob.name.endswith(".pdf"):
                    blob_client = self.container_client.get_blob_client(blob)
                    blob_data = blob_client.download_blob()
                    pages_data.extend(self.read_text_and_metadata_from_pdf(blob_data))
                else:
                    raise ValueError('File format not recognized')

        # if from_blob:
        #     blob_list = self.container_client.list_blobs()
        #     for blob in blob_list:
        #         if blob.name.endswith(".pdf"):
        #             blob_client = self.container_client.get_blob_client(blob)
        #             blob_data = blob_client.download_blob()
        #             pages_data.extend(self.read_text_and_metadata_from_pdf(blob_data))
        #         else:
        #             raise ValueError('File format not recognized')
        # else:
        #     if suffix is None:
        #         raise ValueError('Please add suffix')
        #
        #     for file_name in os.listdir(suffix):
        #         full_path = os.path.join(suffix, file_name)
        #         pages_data.extend(self.read_text_and_metadata_from_pdf_with_meta(full_path))

        return pages_data

    @staticmethod
    def read_text_from_folder() -> str:
        """
        Reads text data from PDF files stored locally in the 'data' folder.

        Returns:
            str: The concatenated text extracted from the PDF files.
        """
        raw_text = ""
        for filename in os.listdir('data'):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join('data', filename)
                pdf_reader = PdfReader(pdf_path)
                for page in pdf_reader.pages:
                    raw_text += page.extract_text()
        return raw_text


class TextPreprocessor:
    @staticmethod
    def get_raw_text(configs) -> str:
        """
        Retrieves raw text based on the specified configuration.

        Args:
            configs: Configuration parameters.

        Returns:
            str: The raw text.
        """
        data_reader = DataReader(configs=configs)
        if configs.storage_type == 'blob':
            raw_text = data_reader.read_text_from_blob_storage()
        elif configs.storage_type == 'local':
            raw_text = data_reader.read_text_from_folder()
        else:
            raise ValueError('Storage_type not recognized')
        return raw_text

    @staticmethod
    def split_text_into_chunks_and_metas(pages_data: List[Tuple[str, dict]],
                                         return_chunks: bool = False,
                                         text_splitter=CharacterTextSplitter):
        """
        Splits the input text into chunks.

        Args:
            pages_data (List[Tuple[str, dict]]): A list of tuples containing text and corresponding metadata.
            return_chunks (bool, optional): Whether to return text chunks or document chunks. Defaults to False.
            text_splitter (type, optional): The type of text splitter to use for chunking. Defaults to
            CharacterTextSplitter.

        Returns:
            Union[List[str], List[Document]]: List of text chunks if return_chunks is False, otherwise list of document
            chunks.

        """
        text_splitter_ = text_splitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        if return_chunks:
            # TODO: OBSERVE THAT THIS APRT HAS TO BE TESTED
            raw_text = "".join(text for text, _ in pages_data)
            return text_splitter_.split_text(raw_text)

        metadatas = [meta_data for text, meta_data in pages_data]
        texts = [text for text, meta_data in pages_data]

        return text_splitter_.create_documents(texts=texts, metadatas=metadatas)

    def split_text_into_chunks_and_metas_wrap(self, configs, return_chunks=False, suffix=None):
        """
          Retrieves text chunks with metadata.

          Args:
              configs : Configuration parameters for the DataReader.
              return_chunks (bool, optional): Whether to return text chunks or document chunks. Defaults to False.
              suffix (str, optional): The directory containing the PDF files if `from_blob` is False. Defaults to None.
              from_blob (bool, optional): Whether to read PDF files from Azure Blob Storage. Defaults to True.

          Returns:
              Union[List[str], List[Document]]: List of text chunks if return_chunks is False, otherwise list of
              document chunks.
          """

        reader = DataReader(configs=configs)

        pages_data = reader.read_text_and_metadata(suffix)

        docs = self.split_text_into_chunks_and_metas(pages_data=pages_data,
                                                     return_chunks=return_chunks,
                                                     text_splitter=configs.text_splitter
                                                     )
        return docs


class PDFConverter:
    def __init__(self, configs):

        blob_service_client = BlobServiceClient.from_connection_string(configs.blob_connection_string)
        container_client = blob_service_client.get_container_client(configs.blob_container_name)
        self.container_client = container_client

    @staticmethod
    def convert_to_pdf_from_local_dir(source_path: str, target_path: str = "data/patents") -> None:
        """
        Convert DOCX files to PDF format and copy PDF files from a local directory to another location.

        Parameters:
            source_path (str): The directory path containing the source files.
            target_path (str, optional): The directory location to save the converted PDF files. Defaults to "data/patents".

        Returns:
            None
        """
        for file in os.listdir(source_path):
            if file.endswith('.docx'):
                name = file.strip('.docx')
                file_path = os.path.join(source_path, file)

                if os.path.exists(f'{target_path}/{name}.pdf'):
                    print('Exists path', f'{target_path}/{name}.pdf')
                    pass
                else:
                    convert(file_path, f'{target_path}/{name}.pdf')

                # convert(file_path, f'{suffix}/{name}.pdf')
            elif file.endswith('.pdf'):
                name = file.strip('.pdf')

                if os.path.exists(f'{target_path}/{name}.pdf'):
                    print('Exists path', f'{target_path}/{name}.pdf')
                    pass
                else:
                    file_path = os.path.join(source_path, file)
                    shutil.copy(file_path, f'{target_path}/{name}.pdf')

            else:
                logging.info(f'Skipping file: {file}')
                pass

    @staticmethod
    def convert_to_pdf_from_docx_bytesio(docx_bytesio, name, target_path="data/patents", temp_file_name='temp.docx'):
        """
         Convert a DOCX file to PDF format.

         Parameters:
             docx_bytesio (BytesIO): BytesIO object containing the DOCX file.
             name (str): Name of the DOCX file.
             target_path (str, optional): Directory location to save the PDF file. Defaults to "data/patents/".
             temp_file_name (str, optional): Temporary file name for conversion. Defaults to 'temp.docx'.
         """

        # Load the docx file
        # doc = Document(docx_bytesio)
        name = name.strip('.docx')

        with open(temp_file_name, 'wb') as f:
            f.write(docx_bytesio.getvalue())

        # Convert the temporary docx file to PDF using docx2pdf
        try:
            convert(temp_file_name, f"{target_path}/{name}.pdf")
        except Exception as e:
            print('Skipping this file, since it is not .docx', e)
            pass

        # Remove the temporary text file
        os.remove(temp_file_name)

    def convert_to_pdf_from_blob(self):
        """
        Convert all DOCX files stored as blobs in a container to PDF format.

        """
        blob_list = self.container_client.list_blobs()
        for blob in blob_list:
            blob_client = self.container_client.get_blob_client(blob)
            blob_data = blob_client.download_blob().readall()

            docx_bytesio = BytesIO(blob_data)

            self.convert_to_pdf_from_docx_bytesio(docx_bytesio, blob.name)

    @staticmethod
    def add_title_to_pdf(pdf_path, title):
        """
         Add a title metadata to a PDF file.

         Parameters:
             pdf_path (str): Path to the PDF file.
             title (str): Title to be added as metadata.
         """
        with open(pdf_path, 'rb') as f:
            pdf_reader = PdfReader(f)
            pdf_writer = PdfWriter()

            # Copy pages from input PDF to output PDF
            for page in pdf_reader.pages:
                pdf_writer.add_page(page)

            # Add metadata (title) to the PDF
            pdf_writer.add_metadata({
                '/Title': title
            })

            # Save the modified PDF with the new title
            with open(pdf_path, 'wb') as output_pdf:
                pdf_writer.write(output_pdf)

    def add_metadata_to_pdf_in_folder(self, target_path='data/patents'):
        """
        Add metadata to all PDF files in a folder.

        Parameters:
            target_path (str, optional): Directory path where PDF files are located. Defaults to 'data/patents'.
        """
        for pdf_path in os.listdir(target_path):
            full_pdf_path = target_path + '/' + pdf_path
            doc_name = str(pdf_path.strip('.pdf'))

            self.add_title_to_pdf(full_pdf_path, doc_name)

    @staticmethod
    def check_if_title_in_pdf(target_path='data/patents'):
        """
        Check if title metadata exists in the PDF files within the specified directory.

        Parameters:
            target_path (str, optional): Directory path where PDF files are located. Defaults to 'data/patents'.
        """
        for pdf_path in os.listdir(target_path):
            pdf_path = target_path + '/' + pdf_path

            with open(pdf_path, 'rb') as f:
                pdf_reader = PdfReader(f)
                meatadata = pdf_reader.metadata
                print(pdf_path, meatadata)

    def execute_main_steps(self, target_path='data/patents', source_path=None):
        if source_path is not None:
            self.convert_to_pdf_from_local_dir(source_path=source_path, target_path=target_path)

        else:
            self.convert_to_pdf_from_blob()

        self.add_metadata_to_pdf_in_folder(target_path=target_path)

        # self.check_if_title_in_pdf(target_path=target_path)


# --------------
# VECTOR STORES
# --------------
def load_data_to_vector_store(configs: Any, return_chunks: bool = False, suffix: str = None) -> None:
    """
    Loads data into the vector store.

    Args:
        configs (Any): The configuration object.
        return_chunks (bool, optional): Whether to return text chunks or document chunks. Defaults to False.
        suffix (str): The directory containing the PDF files if `from_blob` is False. Defaults to None.

    Returns:
        None

    Raises:
        Any exceptions raised during the process.

    """
    text_preprocessor = TextPreprocessor()

    docs = text_preprocessor.split_text_into_chunks_and_metas_wrap(return_chunks=return_chunks,
                                                                   configs=configs,
                                                                   suffix=suffix)

    embeddings = get_embeddings(configs)

    # 1.1 Load the PGVector
    PGVector.from_documents(
        embedding=embeddings,
        documents=docs,
        collection_name=configs.pg_collection_name,
        connection_string=configs.pg_connection_string,
        pre_delete_collection=True)  # To override existent tables with the same name

    logging.info('Documents successfully loaded to vectorstore')


def initialize_pg_vector(configs: Any, return_retriever: bool = True):
    """
    Initialize and return a PGVector vector store or its retriever.

    Args:
        configs (ChatbotConfigs): The configuration object containing PGVector settings.
        return_retriever (bool, optional): Whether to return the retriever instead of the vector store. Defaults to
        True.

    Returns:
        Union[PGVector, emb_utils.Retriever]: The initialized PGVector vector store or its retriever.
    """

    # Load the PGVector vector store
    logging.info('Initializing vectors from vector store')
    embeddings = get_embeddings(configs.llm_configs)

    vec_store = PGVector(
        collection_name=configs.database_configs.pg_collection_name,
        connection_string=configs.database_configs.pg_connection_string,
        embedding_function=embeddings,
    )
    if return_retriever:
        return vec_store.as_retriever()

    return vec_store


def get_embeddings(configs):
    """
    Obtains embeddings based on the specified configuration file.

    Args:
        configs: Configuration parameters.
    Returns:
        Embeddings: Embeddings based on the specified configuration.
    """
    if configs.embeddings_type == "azure_openai":
        return AzureOpenAIEmbeddings(
            azure_deployment=configs.embeddings_deployment,
            openai_api_version=configs.openai_api_version
        )
    elif configs.embeddings_type == "hugging_face":
        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
        return embeddings
    else:
        raise ValueError('Embedding type is not recognized')


def get_vectorstore(configs: conf_.Cfg.llm_configs, text_chunks: List[str], return_embeddings: bool = False):
    """
    Creates a vector store based on the specified configuration and text chunks.

    Args:
        configs (ChatbotConfigs): Configuration parameters.
        text_chunks (List[str]): List of text chunks.
        return_embeddings (bool, optional): Whether to return embeddings along with the vector store.
            Defaults to False.

    Returns:
        FAISS: The created vector store.
    """
    embeddings = get_embeddings(configs=configs)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

    if return_embeddings:
        return vectorstore, embeddings

    return vectorstore
