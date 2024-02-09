import os
import os.path as osp
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, download_url, Data
from typing import Callable, List, Optional

class HeteroDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.root = root
        self.name = name.lower()
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return f'{self.name}.npz'

    @property
    def processed_file_names(self):
        return 'data.pt'

    def process(self):
        data = np.load(self.raw_paths[0])
        x = torch.tensor(data['node_features'])
        y = torch.tensor(data['node_labels'])
        edge_index = torch.tensor(data['edges']).T
        train_mask = torch.tensor(data['train_masks']).to(torch.bool)
        val_mask = torch.tensor(data['val_masks']).to(torch.bool)
        test_mask = torch.tensor(data['test_masks']).to(torch.bool)
        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                        val_mask=val_mask, test_mask=test_mask)
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])


# class LINKXDataset(InMemoryDataset):
#     r"""A variety of non-homophilous graph datasets from the `"Large Scale
#     Learning on Non-Homophilous Graphs: New Benchmarks and Strong Simple
#     Methods" <https://arxiv.org/abs/2110.14446>`_ paper.

#     .. note::
#         Some of the datasets provided in :class:`LINKXDataset` are from other
#         sources, but have been updated with new features and/or labels.

#     Args:
#         root (string): Root directory where the dataset should be saved.
#         name (string): The name of the dataset (:obj:`"penn94"`,
#             :obj:`"reed98"`, :obj:`"amherst41"`, :obj:`"cornell5"`,
#             :obj:`"johnshopkins55"`, :obj:`"genius"`).
#         transform (callable, optional): A function/transform that takes in an
#             :obj:`torch_geometric.data.Data` object and returns a transformed
#             version. The data object will be transformed before every access.
#             (default: :obj:`None`)
#         pre_transform (callable, optional): A function/transform that takes in
#             an :obj:`torch_geometric.data.Data` object and returns a
#             transformed version. The data object will be transformed before
#             being saved to disk. (default: :obj:`None`)
#     """

#     github_url = ('https://github.com/CUAI/Non-Homophily-Large-Scale/'
#                   'raw/master/data')
#     gdrive_url = 'https://drive.google.com/uc?confirm=t&'

#     facebook_datasets = [
#         'penn94', 'reed98', 'amherst41', 'cornell5', 'johnshopkins55'
#     ]

#     datasets = {
#         'penn94': {
#             'Penn94.mat': f'{github_url}/facebook100/Penn94.mat'
#         },
#         'reed98': {
#             'Reed98.mat': f'{github_url}/facebook100/Reed98.mat'
#         },
#         'amherst41': {
#             'Amherst41.mat': f'{github_url}/facebook100/Amherst41.mat',
#         },
#         'cornell5': {
#             'Cornell5.mat': f'{github_url}/facebook100/Cornell5.mat'
#         },
#         'johnshopkins55': {
#             'Johns%20Hopkins55.mat': f'{github_url}/facebook100/Johns%20Hopkins55.mat'
#         },
#         'genius': {
#             'genius.mat': f'{github_url}/genius.mat'
#         },
#         'snap-patents': {
#             'snap-patents.mat': f'{github_url}/snap-patents.mat'
#         },
#         'twitch-de': {
#             'musae_DE_edges.csv': f'{github_url}/twitch/DE/musae_DE_edges.csv',
#             'musae_DE_features.json': f'{github_url}/twitch/DE/musae_DE_features.json',
#             'musae_DE_target.csv': f'{github_url}/twitch/DE/musae_DE_target.csv'
#         },
#         'deezer-europe': {
#             'deezer-europe.mat': f'{github_url}/deezer-europe.mat'
#         },
#         'wiki': {
#             'wiki_views2M.pt':
#             f'{gdrive_url}id=1p5DlVHrnFgYm3VsNIzahSsvCD424AyvP',
#             'wiki_edges2M.pt':
#             f'{gdrive_url}id=14X7FlkjrlUgmnsYtPwdh-gGuFla4yb5u',
#             'wiki_features2M.pt':
#             f'{gdrive_url}id=1ySNspxbK-snNoAZM7oxiWGvOnTRdSyEK'
#         },
#         'twitch-gamer': {
#             'twitch-gamer_feat.csv':
#             f'{gdrive_url}id=1fA9VIIEI8N0L27MSQfcBzJgRQLvSbrvR',
#             'twitch-gamer_edges.csv':
#             f'{gdrive_url}id=1XLETC6dG3lVl7kDmytEJ52hvDMVdxnZ0',
#         },
#         'pokec': {
#             'pokec.mat': f'{github_url}/pokec.mat'
#         },
#         'arxiv-year': {},
#     }

#     splits = {
#         'penn94': f'{github_url}/splits/fb100-Penn94-splits.npy',
#         'genius': f'{github_url}/splits/genius-splits.npy',
#         'twitch-gamers': f'{github_url}/splits/twitch-gamers-splits.npy',
#         'snap-patents': f'{github_url}/splits/snap-patents-splits.npy',
#         'pokec': f'{github_url}/splits/pokec-splits.npy', ### bug to be fixed
#         'twitch-de': f'{github_url}/splits/twitch-e-DE-splits.npy',
#         'deezer-europe': f'{github_url}/splits/deezer-europe-splits.npy',
#     }

#     def __init__(self, root: str, name: str,
#                  transform: Optional[Callable] = None,
#                  pre_transform: Optional[Callable] = None):
#         self.name = name.lower()
#         assert self.name in self.datasets.keys()
#         super().__init__(root, transform, pre_transform)
#         self.data, self.slices = torch.load(self.processed_paths[0])

#     @property
#     def raw_dir(self) -> str:
#         return osp.join(self.root, self.name, 'raw')

#     @property
#     def processed_dir(self) -> str:
#         return osp.join(self.root, self.name, 'processed')

#     @property
#     def raw_file_names(self) -> List[str]:
#         names = list(self.datasets[self.name].keys())
#         if self.name in self.splits:
#             names += [self.splits[self.name].split('/')[-1]]
#         return names

#     @property
#     def processed_file_names(self) -> str:
#         return 'data.pt'

#     def download(self):
#         if self.name in self.datasets.keys():
#             for filename, path in self.datasets[self.name].items():
#                 download_url(path, self.raw_dir, filename=filename)
#         if self.name in self.splits:
#             download_url(self.splits[self.name], self.raw_dir)

#     def _process_wiki(self):
#         paths = {x.split('/')[-1]: x for x in self.raw_paths}
#         x = torch.load(paths['wiki_features2M.pt'])
#         edge_index = torch.load(paths['wiki_edges2M.pt']).t().contiguous()
#         y = torch.load(paths['wiki_views2M.pt'])

#         return Data(x=x, edge_index=edge_index, y=y)

#     def _process_gamer(self):
#         task = 'mature'
#         normalize = True
#         paths = {x.split('/')[-1]: x for x in self.raw_paths}
#         edges = pd.read_csv(paths['twitch-gamer_edges.csv'])
#         nodes = pd.read_csv(paths['twitch-gamer_feat.csv'])
#         edge_index = torch.tensor(edges.to_numpy()).t().type(torch.LongTensor)
#         num_nodes = len(nodes)
#         label, features = load_twitch_gamer(nodes, task)
#         node_feat = torch.tensor(features, dtype=torch.float)
#         if normalize:
#             node_feat = node_feat - node_feat.mean(dim=0, keepdim=True)
#             node_feat = node_feat / node_feat.std(dim=0, keepdim=True)

#         x = node_feat.to(torch.float)
#         y = torch.tensor(label).to(torch.long)
#         edge_index = edge_index.to(torch.long)
#         data = Data(x=x, edge_index=edge_index, y=y)
#         if self.name in self.splits:
#             splits = np.load(paths['twitch-gamer-splits.npy'], allow_pickle=True)
#             sizes = (data.num_nodes, len(splits))
#             data.train_mask = torch.zeros(sizes, dtype=torch.bool)
#             data.val_mask = torch.zeros(sizes, dtype=torch.bool)
#             data.test_mask = torch.zeros(sizes, dtype=torch.bool)

#             for i, split in enumerate(splits):
#                 data.train_mask[:, i][torch.tensor(split['train'])] = True
#                 data.val_mask[:, i][torch.tensor(split['valid'])] = True
#                 data.test_mask[:, i][torch.tensor(split['test'])] = True

#         return data

#     def _process_deezer_europe(self):
#         mat = loadmat(self.raw_paths[0])
#         A, label, features = mat['A'], mat['label'], mat['features']
#         edge_index = torch.from_numpy(np.array(A.nonzero())).to(torch.long)
#         x = torch.tensor(features.todense(), dtype=torch.float)
#         y = torch.tensor(label, dtype=torch.long).squeeze()
#         data = Data(x=x, edge_index=edge_index, y=y)
#         splits = np.load(self.raw_paths[1], allow_pickle=True)
#         sizes = (data.num_nodes, len(splits))
#         data.train_mask = torch.zeros(sizes, dtype=torch.bool)
#         data.val_mask = torch.zeros(sizes, dtype=torch.bool)
#         data.test_mask = torch.zeros(sizes, dtype=torch.bool)
#         for i, split in enumerate(splits):
#             data.train_mask[:, i][torch.tensor(split['train'])] = True
#             data.val_mask[:, i][torch.tensor(split['valid'])] = True
#             data.test_mask[:, i][torch.tensor(split['test'])] = True
#         return data

#     def _process_facebook(self):
#         mat = loadmat(self.raw_paths[0])

#         A = mat['A'].tocsr().tocoo()
#         row = torch.from_numpy(A.row).to(torch.long)
#         col = torch.from_numpy(A.col).to(torch.long)
#         edge_index = torch.stack([row, col], dim=0)

#         metadata = torch.from_numpy(mat['local_info'].astype('int64'))

#         xs = []
#         y = metadata[:, 1] - 1  # gender label, -1 means unlabeled
#         x = torch.cat([metadata[:, :1], metadata[:, 2:]], dim=-1)
#         for i in range(x.size(1)):
#             _, out = x[:, i].unique(return_inverse=True)
#             xs.append(F.one_hot(out).to(torch.float))
#         x = torch.cat(xs, dim=-1)

#         data = Data(x=x, edge_index=edge_index, y=y)
#         if self.name in self.splits:
#             splits = np.load(self.raw_paths[1], allow_pickle=True)
#             sizes = (data.num_nodes, len(splits))
#             data.train_mask = torch.zeros(sizes, dtype=torch.bool)
#             data.val_mask = torch.zeros(sizes, dtype=torch.bool)
#             data.test_mask = torch.zeros(sizes, dtype=torch.bool)

#             for i, split in enumerate(splits):
#                 data.train_mask[:, i][torch.tensor(split['train'])] = True
#                 data.val_mask[:, i][torch.tensor(split['valid'])] = True
#                 data.test_mask[:, i][torch.tensor(split['test'])] = True

#         return data

#     def _process_mat(self):
#         mat = loadmat(self.raw_paths[0])
#         edge_index = torch.from_numpy(mat['edge_index']).to(torch.long)
#         x = torch.from_numpy(mat['node_feat']).to(torch.float)
#         y = torch.from_numpy(mat['label']).squeeze().to(torch.long)
#         data = Data(x=x, edge_index=edge_index, y=y)
#         splits = np.load(self.raw_paths[1], allow_pickle=True)
#         sizes = (data.num_nodes, len(splits))
#         data.train_mask = torch.zeros(sizes, dtype=torch.bool)
#         data.val_mask = torch.zeros(sizes, dtype=torch.bool)
#         data.test_mask = torch.zeros(sizes, dtype=torch.bool)
#         for i, split in enumerate(splits):
#             data.train_mask[:, i][torch.tensor(split['train'])] = True
#             data.val_mask[:, i][torch.tensor(split['valid'])] = True
#             data.test_mask[:, i][torch.tensor(split['test'])] = True
#         return data

#     def _process_snap_patents(self):
#         mat = loadmat(self.raw_paths[0])
#         edge_index = torch.from_numpy(mat['edge_index']).to(torch.long)
#         x = torch.from_numpy(mat['node_feat'].todense()).to(torch.float)
#         years = mat['years'].flatten()
#         nclass = 5
#         labels = even_quantile_labels(years, nclass, verbose=False)
#         y = torch.tensor(labels, dtype=torch.long)

#         data = Data(x=x, edge_index=edge_index, y=y)
#         splits = np.load(self.raw_paths[1], allow_pickle=True)
#         sizes = (data.num_nodes, len(splits))
#         data.train_mask = torch.zeros(sizes, dtype=torch.bool)
#         data.val_mask = torch.zeros(sizes, dtype=torch.bool)
#         data.test_mask = torch.zeros(sizes, dtype=torch.bool)
#         for i, split in enumerate(splits):
#             data.train_mask[:, i][torch.tensor(split['train'])] = True
#             data.val_mask[:, i][torch.tensor(split['valid'])] = True
#             data.test_mask[:, i][torch.tensor(split['test'])] = True
#         return data

#     def _process_twitch_de(self):
#         lang = 'DE'
#         A, label, features = load_twitch(lang, root=self.root)
#         edge_index = torch.from_numpy(np.array(A.nonzero())).to(torch.long)
#         node_feat = torch.tensor(features, dtype=torch.float)
#         label = torch.tensor(label, dtype=torch.long)

#         data = Data(x=node_feat, edge_index=edge_index, y=label)
#         paths = {x.split('/')[-1]: x for x in self.raw_paths}
#         splits = np.load(paths['twitch-e-DE-splits.npy'], allow_pickle=True)
#         sizes = (data.num_nodes, len(splits))
#         data.train_mask = torch.zeros(sizes, dtype=torch.bool)
#         data.val_mask = torch.zeros(sizes, dtype=torch.bool)
#         data.test_mask = torch.zeros(sizes, dtype=torch.bool)
#         for i, split in enumerate(splits):
#             data.train_mask[:, i][torch.tensor(split['train'])] = True
#             data.val_mask[:, i][torch.tensor(split['valid'])] = True
#             data.test_mask[:, i][torch.tensor(split['test'])] = True
#         return data

#     def _process_arxiv_year(self):
#         dataset = NodePropPredDataset(root=self.root, name='ogbn-arxiv')
#         split_path = self.root + "/ogbn_arxiv/arxiv-year-splits.npy"
#         edge_index = torch.as_tensor(dataset.graph['edge_index'])
#         x = torch.as_tensor(dataset.graph['node_feat'])
#         nclass = 5
#         label = even_quantile_labels(
#             dataset.graph['node_year'].flatten(), nclass, verbose=False)
#         y = torch.as_tensor(label).reshape(-1, 1)
#         data = Data(x=x, edge_index=edge_index, y=y)
#         splits = np.load(split_path, allow_pickle=True)
#         sizes = (data.num_nodes, len(splits))
#         data.train_mask = torch.zeros(sizes, dtype=torch.bool)
#         data.val_mask = torch.zeros(sizes, dtype=torch.bool)
#         data.test_mask = torch.zeros(sizes, dtype=torch.bool)
#         for i, split in enumerate(splits):
#             data.train_mask[:, i][torch.tensor(split['train'])] = True
#             data.val_mask[:, i][torch.tensor(split['valid'])] = True
#             data.test_mask[:, i][torch.tensor(split['test'])] = True
#         return data

#     def process(self):
#         if self.name in self.facebook_datasets:
#             data = self._process_facebook()
#         elif self.name in ['genius', 'pokec']:
#             data = self._process_mat()
#         elif self.name == 'deezer-europe':
#             data = self._process_deezer_europe()
#         elif self.name == 'twitch-gamer':
#             data = self._process_gamer()
#         elif self.name == 'wiki':
#             data = self._process_wiki()
#         elif self.name == 'arxiv-year':
#             data = self._process_arxiv_year()
#         elif self.name == 'snap-patents':
#             data = self._process_snap_patents()
#         elif self.name == 'twitch-de':
#             data = self._process_twitch_de()
#         else:
#             raise NotImplementederror(
#                 f"chosen dataset '{self.name}' is not implemented")

#         if self.pre_transform is not None:
#             data = self.pre_transform(data)

#         torch.save(self.collate([data]), self.processed_paths[0])

#     def __repr__(self) -> str:
#         return f'{self.name.capitalize()}({len(self)})'

