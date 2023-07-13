import gzip
import itertools
import json
import logging
import os
import os.path as osp
import shutil
import tarfile
from io import BytesIO
from typing import (Any, Callable, Dict, Iterator, List, Literal, Optional,
                    Sequence, Tuple, Union)

import numpy as np
import pandas as pd
import requests
import torch
from joblib import Parallel, delayed
from ogb.utils import smiles2graph
from rdkit import Chem
from torch import int64
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_gz, extract_zip)
from tqdm.auto import tqdm
from graphgym.utils import grouper


FILLVALUE = "N/A"

OGB_MOL_DATASET_MAP: List[str] = [
    "molbace",
    "molbbbp",
    "molclintox",
    "molmuv",
    "molpcba",
    "molsider",
    "moltox21",
    "moltoxcast",
    "molhiv",
    "molesol",
    "molfreesolv",
]


class OpenMolGraph(InMemoryDataset):
    r"""The Open Moelcular Graph dataset.

    This is a collection of openly accessable molecular graph databases. The
    dataset do **NOT** come with labels. It is intended to be used for
    pre-training and self-suprvised learning.

    Args:
        root: Root directory where the dataset should be saved.
        name: Name of the dataset to use.
        transform: A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
        pre_transform: A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before being saved to
            disk.
        pre_filter: A function that takes in an :obj:`torch_geometric.data.Data`
            object and returns a boolean value, indicating whether the data
            object should be included in the final dataset.
        pre_filter_smiles: A function that determines whether a given SMILES
            should be included. This filtering will be done prior to calling
            pre_filter. Can also pass the name of a pre-defined filtering
            function as a string. Currently supported pre-defined filtering
            functions are: ["organic_ha_only"].
        n_jobs: Number of joblib workers to use when converting SMILES strings
            into graphs.
        batch_size: Number of SMILES to process by each process in each
            iteration.
        joblib_kwargs: Keyword arguments for joblib parallelism when converting
            SMILES strings into graphs.

    """

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        pre_filter_smiles: Optional[Union[str, Callable[[str], bool]]] = None,
        n_jobs: int = 1,
        batch_size: int = 5000,
        joblib_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # TODO: make option to use multiple dataset at once
        # TODO: make subset option
        self.name = name
        self.pre_filter_smiles = pre_filter_smiles

        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.joblib_kwargs = joblib_kwargs or {}

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({len(self):,}, name={self.name!r})"

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str):
        if not isinstance(name, str):
            raise TypeError(f"Name must be a string, got {type(name)}")
        name = name.lower()
        if name in OGB_MOL_DATASET_MAP:
            self.download = self._donwload_ogb_mol
            self.get_smiles_stream = self._stream_ogb_mol_smiles
            self._raw_file_names = [name[3:]]  # strip leading "mol"
        elif name == "geom":
            self.download = self._donwload_geom
            self.get_smiles_stream = self._stream_geom_smiles
            self._raw_file_names = ["summary_drugs.json", "summary_qm9.json"]
        elif name == "chembl":
            self.download = self._donwload_chembl
            self.get_smiles_stream = self._stream_chembl_smiles
            self._raw_file_names = "chembl_32_chemreps.txt"
        elif name == "pubchem10m":
            self.download = self._donwload_pubchem10m
            self.get_smiles_stream = self._stream_pubchem10m_smiles
            self._raw_file_names = [
                "train-00000-of-00001-e9b227f8c7259c8b.parquet",
                "validation-00000-of-00001-9368b7243ba1bff8.parquet"]
        elif name == "zinc20":
            self.download = self._donwload_zinc20
            self.get_smiles_stream = self._stream_zinc20_smiles
            self._raw_file_names = [f"smiles_all_{i:0>2}_clean.jsonl.gz"
                                    for i in range(100)]
        else:
            raise ValueError(f"Unrecognized dataset name {name!r}")
        self._name = name

    @property
    def pre_filter_smiles(self) -> Optional[Callable[[str], bool]]:
        return self._pre_filter_smiles

    @pre_filter_smiles.setter
    def pre_filter_smiles(self, val: Optional[Union[str, Callable]]):
        if isinstance(val, str):
            val = get_predefined_smiles_filter(val)
        self._pre_filter_smiles = val

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.__class__.__name__, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.__class__.__name__, self.name,
                        "processed")

    @property
    def raw_file_names(self) -> List[str]:
        return self._raw_file_names

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def process(self):
        smiles_stream, num_tot_smiles = self.get_smiles_stream()
        batched_smiles_stream = grouper(smiles_stream, n=self.batch_size,
                                        fillvalue=FILLVALUE)
        pbar = tqdm(batched_smiles_stream,
                    total=int(np.ceil(num_tot_smiles / self.batch_size)),
                    desc=f"Processing {num_tot_smiles:,} SMILES")

        # Process SMILES in batches
        parallel = Parallel(n_jobs=self.n_jobs, **self.joblib_kwargs)
        func = delayed(self._batched_smiles_to_data)
        batched_data_lists = parallel(func(smiles, self.pre_filter_smiles)
                                      for smiles in pbar)

        # Combine batches and remove None's
        data_list = list(filter(None, itertools.chain(*batched_data_lists)))
        del batched_data_lists  # release mem

        if (num_filtered := num_tot_smiles - len(data_list)) > 0:
            print(f"Filtered out {num_filtered:,} smiles.")

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        torch.save(self.collate(data_list), self.processed_paths[0])

    @staticmethod
    def _batched_smiles_to_data(
        smiles_seq: Sequence[str],
        pre_filter_smiles: Optional[Callable[[str], bool]],
    ) -> List[Optional[Data]]:
        batched_data_list = [
            OpenMolGraph._smiles_to_data(smiles, pre_filter_smiles)
            for smiles in smiles_seq]
        return batched_data_list

    @staticmethod
    def _smiles_to_data(
        smiles: str,
        pre_filter_smiles: Optional[Callable[[str], bool]],
    ) -> Optional[Data]:
        r"""Convert SMILES into a PyG graph data object using the OGB utils.

        Adapated and modified from the OGB-LSC PCQM4Mv2 processing code:
        https://github.com/snap-stanford/ogb/blob/1.3.6/ogb/lsc/pcqm4mv2_pyg.py

        """
        if smiles == FILLVALUE:
            return None

        if pre_filter_smiles is not None and not pre_filter_smiles(smiles):
            # print(f"Filtering out: {smiles}")
            return None

        try:
            graph = smiles2graph(smiles)
        except Exception as e:
            logging.warning(f"Skipping invalid smiles {smiles!r}: {e}")
            return None

        assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
        assert len(graph["node_feat"]) == graph["num_nodes"]

        data = Data()
        data.__num_nodes__ = int(graph["num_nodes"])
        data.edge_index = torch.from_numpy(graph["edge_index"]).to(int64)
        data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(int64)
        data.x = torch.from_numpy(graph["node_feat"]).to(int64)

        return data

    def _donwload_ogb_mol(self):
        name = self.raw_file_names[0]
        url = f"http://snap.stanford.edu/ogb/data/graphproppred/csv_mol_download/{name}.zip"
        path = download_url(url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def _stream_ogb_mol_smiles(self) -> Tuple[Iterator[str], int]:
        name = self.raw_file_names[0]
        path = osp.join(self.raw_dir, name, "mapping", "mol.csv.gz")
        df = pd.read_csv(path)
        return (i for i in df["smiles"]), len(df)

    def _donwload_geom(self):
        fileid = "4327252"
        url = f"https://dataverse.harvard.edu/api/access/datafile/{fileid}?gbrecs=true"
        download_path = stream_download(url, self.raw_dir)
        with tarfile.open(download_path) as f:
            for file in self.raw_file_names:
                print(f"Extracting {file}")
                file_path = osp.join("rdkit_folder", file)
                f.extract(file_path, path=self.raw_dir)
                os.rename(osp.join(self.raw_dir, file_path),
                          osp.join(self.raw_dir, file))
            shutil.rmtree(osp.join(self.raw_dir, "rdkit_folder"))
        os.unlink(download_path)

    def _stream_geom_smiles(self) -> Tuple[Iterator[str], int]:
        smiles = set()
        for path in self.raw_paths:
            with open(path) as f:
                smiles.update(list(json.load(f)))
        smiles = sorted(smiles)
        return smiles, len(smiles)

    def _donwload_chembl(self):
        url = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_32/chembl_32_chemreps.txt.gz"
        download_path = download_url(url, self.raw_dir)
        extract_gz(download_path, self.raw_dir)
        os.unlink(download_path)

    def _stream_chembl_smiles(self) -> Tuple[Iterator[str], int]:
        df = pd.read_csv(self.raw_paths[0], sep="\t")
        return (i for i in df["canonical_smiles"]), len(df)

    def _donwload_pubchem10m(self):
        base_url = "https://huggingface.co/datasets/sagawa/pubchem-10m-canonicalized/resolve/main/data"
        for file in self.raw_file_names:
            download_url(f"{base_url}/{file}", self.raw_dir)

    def _stream_pubchem10m_smiles(self) -> Tuple[Iterator[str], int]:
        df = pd.concat([pd.read_parquet(path, engine="pyarrow")
                        for path in self.raw_paths])
        return (i for i in df["smiles"]), len(df)

    def _donwload_zinc20(self):
        processed = os.listdir(self.raw_dir) if osp.isdir(self.raw_dir) else []
        num_total_files = len(self.raw_file_names)
        for i, file in enumerate(self.raw_file_names):
            if file in processed:
                continue
            print(f"File {i + 1} / {num_total_files}")
            url = f"https://huggingface.co/datasets/zpn/zinc20/resolve/main/zinc_processed/{file}"
            download_url(url, self.raw_dir)

    def _stream_zinc20_smiles(self) -> Tuple[Iterator[str], int]:
        return stream_zpn_smiles(self.raw_paths, mode="zinc20",
                                 num_tot_smiles=1_006_650_595)


def stream_zpn_smiles(
    paths: List[str],
    *,
    mode: Literal["zinc20", "pubchem"],
    num_tot_smiles: Optional[int] = None,
) -> Tuple[Iterator[str], int]:
    if num_tot_smiles is None:
        num_tot_smiles = 0
        for path in paths:
            print(f"Counting {path}")
            with gzip.open(path, "rb") as f:
                for _ in f:
                    num_tot_smiles += 1

    def smiles_streamer() -> Iterator[str]:
        for path in paths:
            for item in stream_jsonl_gz(path):
                if mode == "zinc20":
                    yield item["smiles"]
                elif mode == "pubchem":
                    yield item["molecules"][0]["properties"]["PUBCHEM_OPENEYE_CAN_SMILES"]
                else:
                    raise ValueError(f"Unknown mode {mode!r}")

    return smiles_streamer(), num_tot_smiles


def stream_jsonl_gz(path: str) -> Iterator[Dict[str, Any]]:
    with gzip.open(path, "rb") as f:
        for i in f:
            yield json.loads(i.strip())


def stream_download(url: str, folder: str) -> str:
    print(f"Downloading {url}")
    r = requests.get(url, stream=True)
    tot_bytes = int(r.headers.get("content-length", 0))
    pbar = tqdm(
        total=tot_bytes,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    )

    with BytesIO() as b:
        for data in r.iter_content(1024):
            pbar.update(len(data))
            b.write(data)
        content = b.getvalue()

    filename = url.rpartition("/")[2]
    filename = filename if filename[0] == "?" else filename.split("?")[0]
    path = osp.join(folder, filename)
    with open(path, "wb") as f:
        f.write(content)

    return path


def get_predefined_smiles_filter(name: str) -> Callable[[str], bool]:
    available_options = ["organic_ha_only"]
    if name == "organic_ha_only":
        return get_smiles_filter_organic_ha_only()
    else:
        raise ValueError(f"Unknown smiles filter name {name!r}. Available "
                         f"options are {available_options}")


def get_smiles_filter_organic_ha_only() -> Callable[[str], bool]:
    """Filter out SMILES that contain atoms other than H, C, N, O, S."""
    inorganic_ha_smarts = Chem.MolFromSmarts("[!#1;!#6;!#7;!#8;!#16]")

    def smiles_filter_organic_ha_only(smiles: str) -> bool:
        mol = Chem.MolFromSmiles(smiles)
        return not (mol is None or mol.HasSubstructMatch(inorganic_ha_smarts))

    return smiles_filter_organic_ha_only


if __name__ == "__main__":
    print(OpenMolGraph("test_omg", "molbace", pre_filter_smiles="organic_ha_only"))
    shutil.rmtree("test_omg")
    # print(OpenMolGraph("/mnt/scratch/liurenmi/test_omg", "molbace"))
    # print(OpenMolGraph("/mnt/scratch/liurenmi/test_omg", "molhiv"))
    # print(OpenMolGraph("/mnt/scratch/liurenmi/test_omg", "chembl", n_jobs=4))
    # print(OpenMolGraph("/mnt/scratch/liurenmi/test_omg", "geom", n_jobs=4))
    # print(OpenMolGraph("/mnt/scratch/liurenmi/test_omg", "pubchem10m"))
    # print(OpenMolGraph("/mnt/scratch/liurenmi/test_omg", name="zinc20",
    #                    n_jobs=128, batch_size=100_000))
    # print(OpenMolGraph("/mnt/scratch/liurenmi/test_omg_filtered", "chembl",
    #                    n_jobs=8, pre_filter_smiles="organic_ha_only"))
    # print(OpenMolGraph("/mnt/scratch/liurenmi/test_omg_filtered", n_jobs=8,
    #                    name="pubchem10m", pre_filter_smiles="organic_ha_only"))
    # print(OpenMolGraph("/mnt/scratch/liurenmi/test_omg_filtered", n_jobs=128,
    #                    name="zinc20", batch_size=100_000,
    #                    pre_filter_smiles="organic_ha_only",
    #                    joblib_kwargs={"pre_dispatch": "all"}))
