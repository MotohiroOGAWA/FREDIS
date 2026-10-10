import os
from typing import Dict, Tuple, List, Literal
import torch
import pandas as pd
from datetime import datetime, timezone
from torch.utils.tensorboard import SummaryWriter

from .model_base import ModelBase
from .training_setup import get_optimizer, get_scheduler
from .logger import Metrics

class CheckPointManager:
    DIR_NAME = 'checkpoints'
    CKPT_TREE_FILE_NAME = 'ckpt_tree.tsv'
    CHECKPOINT_TREE_COLUMNS = ['id', 'branch_id', 'name', 'epoch', 'iter', 'parent_id', 'comment']
    TOPK_TABLE_COLUMNS = ['topk_id', 'branch_id', 'prev_ckpt_id', 'name', 'epoch', 'iter', 'score', 'comment']
    CHECKPOINT_TREE_DTYPES = {
        'id': 'int64',
        'branch_id': 'int64',
        'name': 'string',
        'epoch': 'Int64', # allow NaN for root nodes
        'iter': 'Int64', # allow NaN for root nodes
        'parent_id': 'int64'
    }
    def __init__(self, ckpt_root_dir: str):
        self.ckpt_root_dir = ckpt_root_dir

        self.ckpt_nodes, self.base_branch_ids = self._read_checkpoint_nodes()

        self._current_branch_id = None

        self._metrics: Metrics = None
        self._topk_df: pd.DataFrame = None
        self._topk_store = {}

        self._run_dir: str = None

        self._summary_writer: SummaryWriter = None

    def _reset(self):
        self._current_branch_id = None
        self._metrics = None
        self._topk_df = None
        self._topk_store = {}
        self._run_dir = None
        self._summary_writer = None

    @property
    def ckpt_dir(self):
        return os.path.join(self.ckpt_root_dir, CheckPointManager.DIR_NAME)
    
    @property
    def current_branch_id(self):
        return self._current_branch_id
    
    @current_branch_id.setter
    def current_branch_id(self, branch_id:int):
        if self._current_branch_id is not None and self._current_branch_id == branch_id:
            return
        else:
            if branch_id is None:
                self._reset()
            else:
                self._init_topk(branch_id=branch_id)
                self._current_branch_id = branch_id

    def checkout_base(self) -> 'CkptNode':
        """
        Create and checkout a new base branch.

        This function creates a new branch starting from scratch 
        (i.e., with no root checkpoint, root_ckpt_node_id = -1) 
        and sets it as the current branch.

        Returns:
            CkptNode: The root node of the newly created base branch.
        """
        new_root_node = self._create_branch(root_ckpt_node_id=-1)
        self.current_branch_id = new_root_node.branch_id
        return new_root_node

    def checkout_latest(self, branch_id:int) -> 'CkptNode':
        """
        Checkout the latest checkpoint from an existing branch.

        This function sets the given branch as the current branch 
        and returns its latest checkpoint node.

        Args:
            branch_id (int): The ID of the branch to checkout.

        Returns:
            CkptNode: The latest checkpoint node of the specified branch.

        Raises:
            ValueError: If the given branch_id does not exist.
        """
        if branch_id not in self.ckpt_nodes:
            raise ValueError(f"Invalid branch ID: {branch_id}")
        latest_ckpt = self._get_latest_checkpoint(branch_id)
        self.current_branch_id = branch_id
        return latest_ckpt
    
    def checkout_new_ckpt(self):
        """
        Create and register a new checkpoint in the current branch.

        If no branch has been selected yet, a new branch will be created first
        and set as the current branch. Then, a new checkpoint is created and 
        added to this branch.

        Returns:
            latest_ckpt: The newly created checkpoint associated with the current branch.
        """
        if self.current_branch_id is None:
            new_branch_node = self.checkout_new_branch()
        
        latest_ckpt = self._create_new_ckpt()
        return latest_ckpt
    
    def checkout_new_branch(self, root_ckpt_node_id: int = -1) -> 'CkptNode':
        """
        Create a new branch and set it as the current branch.

        If a valid root checkpoint node ID is provided, the new branch will be
        created starting from that checkpoint. Otherwise (root_ckpt_node_id = -1),
        the branch will be created as a fresh branch without a specified root.
        After creation, the current branch pointer is updated to this new branch.

        Args:
            root_ckpt_node_id (int, optional): ID of the root checkpoint node 
                from which the new branch should originate. Default is -1 
                (no specific root).

        Returns:
            CkptNode: The root node of the newly created branch.
        """
        assert root_ckpt_node_id == -1 or (root_ckpt_node_id in self.ckpt_nodes), "Invalid root checkpoint node ID."
        new_branch_node = self._create_branch(root_ckpt_node_id=root_ckpt_node_id)
        self.current_branch_id = new_branch_node.branch_id
        return new_branch_node

    def _create_new_ckpt(self) -> 'CkptNode':
        assert self.current_branch_id is not None, "No current branch. Please checkout a branch first."

        pre_ckpt_node = self._get_latest_checkpoint(self.current_branch_id)
        new_ckpt_id = self._get_new_ckpt_id()
        new_ckpt_node = CkptNode(
            root_dir=self.ckpt_dir,
            id=new_ckpt_id,
            branch_id=self.current_branch_id,
            name='',
            epoch=None,
            iter=None,
            parent_id=pre_ckpt_node.id,
            next_id=None,
        )
        self.ckpt_nodes[pre_ckpt_node.id].next_id = new_ckpt_id
        self.ckpt_nodes[new_ckpt_id] = new_ckpt_node
        return new_ckpt_node

    def _create_branch(self, root_ckpt_node_id: int = -1) -> 'CkptNode':
        new_branch_id = self._get_new_branch_id()
        if root_ckpt_node_id != -1:
            if root_ckpt_node_id not in self.ckpt_nodes:
                raise ValueError(f"Invalid checkpoint node ID: {root_ckpt_node_id}")
            # if not CkptNode.is_ckpt_node(root_ckpt_node_id):
            #     raise ValueError(f"Invalid checkpoint node ID: {root_ckpt_node_id}")
        
        new_branch_node = CkptNode(
            root_dir=self.ckpt_dir,
            id=new_branch_id,
            branch_id=new_branch_id,
            name='',
            epoch=None,
            iter=None,
            parent_id=root_ckpt_node_id,
            next_id=None,
        )
        self.ckpt_nodes[new_branch_id] = new_branch_node

        if root_ckpt_node_id == -1:
            self.base_branch_ids.append(new_branch_id)
        else:
            self.ckpt_nodes[root_ckpt_node_id].add_child(new_branch_id)

        return new_branch_node

    def _get_latest_checkpoint(self, branch_id: int) -> 'CkptNode':
        # Find the latest checkpoint for the given branch ID
        next_node = self.ckpt_nodes[branch_id]
        while not next_node.is_latest:
            next_node = self.ckpt_nodes[next_node.next_id]
        return next_node

    def _read_checkpoint_nodes(self) -> Tuple[Dict[int, 'CkptNode'], List[int]]:
        meta_path = os.path.join(self.ckpt_dir, CheckPointManager.CKPT_TREE_FILE_NAME)
        if not os.path.exists(meta_path):
            df = pd.DataFrame(columns=CheckPointManager.CHECKPOINT_TREE_COLUMNS)
            df = df.astype(CheckPointManager.CHECKPOINT_TREE_DTYPES)
        else:
            df = pd.read_csv(meta_path, sep="\t", header=0, dtype=CheckPointManager.CHECKPOINT_TREE_DTYPES)


        ckpt_nodes = {row['id']: CkptNode(
            root_dir=self.ckpt_dir,
            id=row['id'],
            branch_id=row['branch_id'],
            name=row['name'],
            epoch=row['epoch'],
            iter=row['iter'],
            parent_id=row['parent_id'],
            next_id=None,
        ) for _, row in df.iterrows()}
        
        base_branch_ids = []
        for _, row in df.iterrows():
            parent_id = row['parent_id']
            if parent_id == -1:
                base_branch_ids.append(row['id'])
            elif parent_id in ckpt_nodes:
                if row['branch_id'] == ckpt_nodes[row['parent_id']].branch_id:
                    ckpt_nodes[row['parent_id']].next_id = row['id']
                else:
                    ckpt_nodes[parent_id].add_child(row['id'])
            else:
                raise ValueError(f"Parent ID {parent_id} not found for checkpoint {row['id']}")

        return ckpt_nodes, base_branch_ids

    def _get_new_branch_id(self) -> int:
        """
        Get a new branch ID.
        """
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        while True:
            new_branch_id = int('1'+str(now_ms))
            if new_branch_id not in self.ckpt_nodes:
                return new_branch_id
            now_ms += 1

    def _get_new_ckpt_id(self) -> int:
        """
        Get a new checkpoint ID.
        """
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        while True:
            new_ckpt_id = int('2'+str(now_ms))
            if new_ckpt_id not in self.ckpt_nodes:
                return new_ckpt_id
            now_ms += 1
            
    def _get_new_topk_id(self) -> int:
        """
        Get a new Top-K ID.
        """
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        while True:
            new_topk_id = int('3'+str(now_ms))
            if new_topk_id not in self._topk_store:
                return new_topk_id
            now_ms += 1


    def get_node_lineage(self, node_id: int, newest_first: bool = True) -> List[int]:
        """
        Trace back from the given node (branch or checkpoint) to the root (-1).
        Collect all node IDs (branch and checkpoint) along the lineage.

        Args:
            node_id (int): Starting node id (can be a branch node or checkpoint node).
            newest_first (bool): If True, return in newest-first order.
                                 If False, return in oldest-first order.

        Returns:
            List[int]: List of node IDs (branch and checkpoint) along the lineage.
        """
        assert node_id in self.ckpt_nodes, f"Invalid node_id {node_id}"
        
        lineage = []
        node = self.ckpt_nodes[node_id]

        while True:
            # Record the current node id (branch or checkpoint)
            lineage.append(node.id)
            
            # Stop if we reached the root
            if node.parent_id == -1:
                break

            # Move to the parent node
            parent_id = node.parent_id
            assert parent_id in self.ckpt_nodes, f"Parent {parent_id} not found"
            node = self.ckpt_nodes[parent_id]

        # Reverse the order if newest-first is requested
        if not newest_first:
            lineage = lineage[::-1]
        return lineage
    
    def get_branch_lineage(self, node_id: int, newest_first: bool = True) -> List[int]:
        """
        Trace back from the given node (branch or checkpoint) to the root (-1).
        Collect all branch_ids along the lineage.

        Args:
            node_id (int): Starting node id (can be a branch node or checkpoint node).
            newest_first (bool): If True, return branch_ids in newest-first order.
                                 If False, return in oldest-first order.

        Returns:
            List[int]: List of branch_ids along the lineage.
        """
        assert node_id in self.ckpt_nodes, f"Invalid node_id {node_id}"
        
        lineage = []
        node = self.ckpt_nodes[node_id]

        while True:
            # Record branch_id if the current node is a branch node
            if CkptNode.is_branch_node(node.id):
                lineage.append(node.branch_id)
            
            # Stop if we reached the root
            if node.parent_id == -1:
                break

            # Move to the parent node
            parent_id = node.parent_id
            assert parent_id in self.ckpt_nodes, f"Parent {parent_id} not found"
            node = self.ckpt_nodes[parent_id]

        # Reverse for newest-first order if requested
        if not newest_first:
            lineage = lineage[::-1]
        return lineage

    def get_all_related_branches(self, branch_id: int) -> Dict[int, dict]:
        """
        Get all branches related to the given branch_id.
        This means:
        - Find the root branch of the lineage
        - Collect all branches under that root (siblings, descendants, etc.)
        - Return as nested dict structure
        
        Args:
            branch_id (int): Branch ID to query
        
        Returns:
            Dict[int, dict]: Nested dict representing branch hierarchy
        """
        assert branch_id in self.ckpt_nodes, f"Invalid branch_id {branch_id}"

        # --- 1. lineage to root (oldest-first order)
        lineage = self.get_branch_lineage(branch_id, newest_first=False)
        root_id = lineage[0]  # the oldest branch_id (root)

        # --- 2. build recursive structure
        def build_branch_dict(node_id: int) -> Dict[int, dict]:
            """
            Recursively build a nested dict of all descendant branches
            starting from the given node (branch or checkpoint).
            """
            branch_dict = {}
            next_id = node_id
            while next_id is not None:
                next_node = self.ckpt_nodes[next_id]
                child_ids = next_node.child_ids
                for c_id in child_ids:
                    if CkptNode.is_branch_node(c_id):
                        branch_dict[c_id] = build_branch_dict(c_id)
                    else:
                        raise NotImplementedError("Checkpoint node structure not supported")
                next_id = next_node.next_id
            return branch_dict

        # --- 3. assemble
        result = {root_id: build_branch_dict(root_id)}
        return result

    def update(self):
        """
        Update the checkpoint tree and synchronize Top-K data.
        - Save checkpoint tree to disk
        - Save Top-K table and models to disk
        - Remove unused Top-K models
        """
        # --- 1. save checkpoint tree ---
        cols ={
            'id': [],
            'branch_id': [],
            'name': [],
            'epoch': [],
            'iter': [],
            'parent_id': [],
            'comment': []
        }

        for ckpt_id, ckpt_node in sorted(self.ckpt_nodes.items(), key=lambda x: x[0]):
            cols['id'].append(ckpt_node.id)
            cols['branch_id'].append(ckpt_node.branch_id)
            cols['name'].append(ckpt_node.name)
            cols['epoch'].append(ckpt_node.epoch)
            cols['iter'].append(ckpt_node.iter)
            cols['parent_id'].append(ckpt_node.parent_id)
            cols['comment'].append(ckpt_node.comment)

        os.makedirs(self.ckpt_dir, exist_ok=True)
        df = pd.DataFrame(cols, columns=self.CHECKPOINT_TREE_COLUMNS)
        df.to_csv(os.path.join(self.ckpt_dir, CheckPointManager.CKPT_TREE_FILE_NAME), sep="\t", index=False)

        # --- 2. save Top-K table and store ---
        if self._topk_df is not None:
            self._save_topk(self.current_branch_id)

        self.ckpt_nodes, self.base_branch_ids = self._read_checkpoint_nodes()

    def load_ckpt(self, ckpt_id:int) -> 'CkptNode':
        assert ckpt_id in self.ckpt_nodes, f"Checkpoint ID {ckpt_id} not found."
        ckpt_node = self.ckpt_nodes[ckpt_id]

        if ckpt_node.is_latest:
            self.current_branch_id = ckpt_node.branch_id
        else:
            new_branch_node = self.checkout_new_branch(root_ckpt_node_id=ckpt_id)
            ckpt_node = new_branch_node

        while ckpt_node.is_branch:
            ckpt_node = self.ckpt_nodes[ckpt_node.parent_id]

        return ckpt_node


    def initialize_metrics(self, columns:List[str]):
        self._metrics = Metrics(columns=columns)

    def log_metrics(self, **kwargs):
        assert self._metrics is not None, "Metrics logger is not initialized."
        self._metrics.log(**kwargs)

    def flush_metrics(self, flush_dir: str = None):
        assert self._metrics is not None, "Metrics logger is not initialized."
        if flush_dir is None:
            ckpt_node = self._get_latest_checkpoint(self.current_branch_id)
            flush_dir = ckpt_node.dir
        self._metrics.flush(os.path.join(flush_dir, 'metrics.tsv'))

    def read_metrics(self, ckpt_id:int = None) -> pd.DataFrame:
        assert ckpt_id is not None or self.current_branch_id is not None, "No current branch. Please checkout a branch first."
        if ckpt_id is None:
            ckpt_node = self._get_latest_checkpoint(self.current_branch_id)
        else:
            ckpt_node = self.ckpt_nodes[ckpt_id]
            if ckpt_node.is_branch:
                ckpt_node = self._get_latest_checkpoint(ckpt_node.branch_id)

        metrics_df_list = []
        while True:
            file_path = os.path.join(ckpt_node.dir, 'metrics.tsv')
            if os.path.exists(file_path):
                metrics_df = pd.read_csv(file_path, sep="\t", header=0)
                metrics_df['ckpt_id'] = ckpt_node.id
                metrics_df_list.append(metrics_df)  
            if ckpt_node.is_base:
                break
            ckpt_node = self.ckpt_nodes[ckpt_node.parent_id]
        metrics_df_list = metrics_df_list[::-1]
        merged_df = pd.concat(metrics_df_list, ignore_index=True)
        merged_df['ckpt_id'] = merged_df['ckpt_id'].astype(int)
        return merged_df
    


    def get_topk_dir(self, branch_id:int):
        assert CkptNode.is_branch_node(branch_id), "Invalid branch ID."
        return os.path.join(self.ckpt_dir, str(branch_id), 'topk')

    def get_topk_file(self, branch_id:int) -> str:
        topk_dir = self.get_topk_dir(branch_id)
        return os.path.join(topk_dir, 'topk.tsv')
    
    def get_topk_store_dir(self, branch_id:int):
        topk_dir = self.get_topk_dir(branch_id)
        topk_store_dir = os.path.join(topk_dir, 'store')
        return topk_store_dir

    def get_topk_df(self, branch_id:int):
        topk_file = self.get_topk_file(branch_id)
        assert os.path.exists(topk_file), "Top-K file does not exist."
        return pd.read_csv(topk_file, sep="\t", header=0)
    
    def _init_topk(self, branch_id:int=None):
        if branch_id is None:
            assert self.current_branch_id is not None, "No current branch. Please checkout a branch first."
            branch_id = self.current_branch_id

        topk_file = self.get_topk_file(branch_id)

        topk_df = None
        valid_branch_id = None
        if os.path.exists(topk_file):
            topk_df = pd.read_csv(topk_file, sep="\t", header=0)
            valid_branch_id = branch_id
        else:
            branch_lineage = self.get_branch_lineage(branch_id)

            for b_id in branch_lineage[1:]:
                _file = self.get_topk_file(b_id)
                if os.path.exists(_file):
                    topk_df = pd.read_csv(_file, sep="\t", header=0)
                    valid_branch_id = b_id
                    break
            else:
                topk_df = pd.DataFrame(columns=CheckPointManager.TOPK_TABLE_COLUMNS)

        latest_ckpt_node = self._get_latest_checkpoint(branch_id)
        node_lineage = self.get_node_lineage(latest_ckpt_node.id)

        # Filter rows: keep only those inheritable via prev_ckpt_id → next_id
        valid_rows = []
        for _, row in topk_df.iterrows():
            prev_id = row["prev_ckpt_id"]

            if pd.isna(prev_id):
                continue
            prev_id = int(prev_id)

            # Check that prev_id exists in ckpt_nodes
            if prev_id not in self.ckpt_nodes:
                continue

            next_id = self.ckpt_nodes[prev_id].next_id
            if next_id is not None and next_id in node_lineage:
                valid_rows.append(row)

        # Create filtered dataframe
        topk_df = pd.DataFrame(valid_rows, columns=CheckPointManager.TOPK_TABLE_COLUMNS)

        topk_store = {}
        if valid_branch_id is not None:
            for topk_id in topk_df['topk_id'].values:
                topk_store[int(topk_id)] = self._load_topk(valid_branch_id, int(topk_id))

        self._topk_df = topk_df
        self._topk_store = topk_store

    def update_topk(self, score, epoch:int, iter:int, model:ModelBase, optimizer: torch.optim.Optimizer = None, scheduler: torch.optim.lr_scheduler._LRScheduler = None, optimizer_info = None, extra_data=None, topk:int=1, comment:str='') -> int:
        if self._topk_df is None:
            self._init_topk()

        # Skip if not good enough
        if len(self._topk_df) >= topk and score > self._topk_df['score'].max():
            return -1

        new_topk_id = self._get_new_topk_id()
        latest_ckpt_node = self._get_latest_checkpoint(self.current_branch_id)

        new_row = {
            'topk_id': new_topk_id,
            'branch_id': self.current_branch_id,
            'prev_ckpt_id': latest_ckpt_node.id,
            'name': '',
            'epoch': epoch,
            'iter': iter,
            'score': score,
            'comment': comment
        }
        data = CkptNode.convert_save_data(epoch, iter, model, optimizer, scheduler, optimizer_info, extra_data)

        # Update DataFrame
        tmp_df = pd.concat([self._topk_df, pd.DataFrame([new_row])], ignore_index=True)
        tmp_df = tmp_df.sort_values("score", ascending=True).head(topk)

        # Update in-memory store (temporary, not saved to disk here)
        tmp_store = {}
        for tid in tmp_df["topk_id"].values:
            if tid == new_topk_id:
                tmp_store[tid] = data  # use passed data for the new entry
            elif tid in self._topk_store:
                tmp_store[tid] = self._topk_store[tid]

        # Replace internal state
        self._topk_df = tmp_df.reset_index(drop=True)
        self._topk_store = tmp_store

        return new_topk_id
    
    def _load_topk(self, branch_id:int, topk_id:int):
        assert branch_id is not None, "Branch ID must be provided."

        topk_dir = self.get_topk_dir(branch_id)
        file = os.path.join(topk_dir, 'store', f"{topk_id}.pt")
        assert os.path.exists(file), f"Top-K model file {file} does not exist."

        return torch.load(file, weights_only=False)

    def _save_topk(self, branch_id: int):
        """
        Save Top-K table and models for the given branch.
        Removes outdated entries.
        """
        # --- save topk.tsv ---
        topk_dir = self.get_topk_dir(branch_id)
        os.makedirs(topk_dir, exist_ok=True)
        self._topk_df.to_csv(self.get_topk_file(branch_id), sep="\t", index=False)

        # --- save store ---
        topk_store_dir = self.get_topk_store_dir(branch_id)
        os.makedirs(topk_store_dir, exist_ok=True)

        keep_ids = set(self._topk_df["topk_id"].values)

        for tid, obj in self._topk_store.items():
            file_path = os.path.join(topk_store_dir, f"{tid}.pt")
            if tid in keep_ids:
                # save if not exists
                if not os.path.exists(file_path):
                    torch.save(obj, file_path)
            else:
                # remove unused file if exists
                if os.path.exists(file_path):
                    os.remove(file_path)

        # --- cleanup stray files ---
        for file in os.listdir(topk_store_dir):
            if file.endswith(".pt"):
                tid = int(file.replace(".pt", ""))
                if tid not in keep_ids:
                    os.remove(os.path.join(topk_store_dir, file))

    def get_topk_models(self, branch_id: int, reload_from_disk: bool = False) -> Tuple[Dict[int, dict], pd.DataFrame]:
        """
        Get Top-K models and their metadata for the given branch lineage.

        Args:
            branch_id (int): Branch ID to fetch Top-K models from.
            reload_from_disk (bool): If True, always reload models from disk.
                                    If False, use in-memory store if available.

        Returns:
            models (Dict[int, dict]): rank (1,2,...) -> model save_data (dict with state_dict, etc.)
            info   (pd.DataFrame): Top-K table rows (filtered by lineage already in _init_topk)
        """
        # Ensure Top-K is initialized
        self.current_branch_id = branch_id

        # Do not filter by branch_id here (lineage already resolved in _init_topk)
        topk_df = self._topk_df.reset_index(drop=True)

        models = {}
        for rank, row in enumerate(topk_df.itertuples(index=False), start=1):
            tid = int(row.topk_id)
            if not CkptNode.is_topk_node(tid):
                raise ValueError(f"Invalid Top-K ID {tid} (row: {row})")

            if not reload_from_disk and tid in self._topk_store:
                models[rank] = self._topk_store[tid]
            else:
                models[rank] = self._load_topk(row.branch_id, tid)

        return models, topk_df

    def get_all_related_topk_models(self, branch_id: int, reload_from_disk: bool = False) -> Tuple[Dict[int, dict], pd.DataFrame]:
        """
        Collect Top-K models from all related branches (same root lineage).
        - Avoid duplicates (same topk_id appears only once)
        - Merge into single ranking table

        Args:
            branch_id (int): Branch ID to start from
            reload_from_disk (bool): If True, reload from disk instead of in-memory

        Returns:
            models (Dict[int, dict]): rank (1,2,...) -> model save_data
            info   (pd.DataFrame): merged Top-K table without duplicate topk_ids
        """
        # --- 1. collect all branch_ids under the same root
        branch_tree = self.get_all_related_branches(branch_id)

        # flatten dict structure -> set of branch_ids
        def flatten_branch_dict(bdict: Dict[int, dict]) -> List[int]:
            ids = []
            for bid, sub in bdict.items():
                ids.append(bid)
                ids.extend(flatten_branch_dict(sub))
            return ids

        related_branch_ids = flatten_branch_dict(branch_tree)

        # --- 2. collect topk from each branch
        seen_ids = set()
        all_rows = []
        all_models = {}

        for b_id in related_branch_ids:
            models, df = self.get_topk_models(b_id, reload_from_disk=reload_from_disk)
            for row in df.itertuples(index=False):
                tid = int(row.topk_id)
                if tid in seen_ids:
                    continue
                seen_ids.add(tid)
                all_rows.append(row._asdict())
            # merge models dict (rank is not preserved across branches, re-rank later)
            for rank, data in models.items():
                tid = df.iloc[rank-1].topk_id
                if tid not in all_models:
                    all_models[tid] = data

        # --- 3. build merged DataFrame
        merged_df = pd.DataFrame(all_rows, columns=CheckPointManager.TOPK_TABLE_COLUMNS)

        # sort by score ascending, reset index → ranking
        merged_df = merged_df.sort_values("score", ascending=True).reset_index(drop=True)

        # --- 4. rebuild models dict using global ranking
        ranked_models = {}
        for rank, row in enumerate(merged_df.itertuples(index=False), start=1):
            tid = int(row.topk_id)
            ranked_models[rank] = all_models[tid]

        return ranked_models, merged_df



    def set_run_dir(self, run_dir: str):
        self._run_dir = run_dir

    def get_run_dir(self, ckpt_id=None) -> str:
        assert self._run_dir is not None, "Run directory is not set. Please set it using set_run_dir()."
        if ckpt_id is None:
            assert self.current_branch_id is not None, "No current branch. Please checkout a branch first."
            ckpt_node = self._get_latest_checkpoint(self.current_branch_id)
        else:
            ckpt_node = self.ckpt_nodes[ckpt_id]
        relative_path = os.path.relpath(self._run_dir, ckpt_node.dir)
        return relative_path

    def flush_run_dir_path(self, flush_dir: str = None):
        assert self._run_dir is not None, "Run directory is not set. Please set it using set_run_dir()."
        if flush_dir is None:
            ckpt_node = self._get_latest_checkpoint(self.current_branch_id)
            flush_dir = ckpt_node.dir
        with open(os.path.join(flush_dir, 'run_dir.txt'), 'w') as f:
            f.write(self.get_run_dir())

    
    @property
    def summary_writer_log_dir(self) -> str:
        assert self._run_dir is not None, "Run directory is not set. Please set it using set_run_dir()."
        return os.path.join(self._run_dir, 'log')
    
    @staticmethod
    def get_tensorboard_command(log_dir:str) -> str:
        return f"tensorboard --logdir {log_dir} --port 6006 --host 0.0.0.0"
    
    @staticmethod
    def get_tensorboard_url() -> str:
        return "http://localhost:6006/"

    @property
    def summary_writer(self) -> SummaryWriter:
        """
        Returns a TensorBoard SummaryWriter instance for logging training/validation metrics.

        - Log files are saved under: <run_dir>/log/
        - To visualize logs in real time, run the following command:
            tensorboard --logdir runs/<run_id>/log --port 6006 --host 0.0.0.0

        Note:
            - Requires that self._run_dir has been set beforehand (via set_run_dir()).
            - The SummaryWriter instance is lazily initialized (created only once).
        """
        assert self._run_dir is not None, "Run directory is not set. Please set it using set_run_dir()."
        if self._summary_writer is None:
            log_dir = self.summary_writer_log_dir
            self._summary_writer = SummaryWriter(log_dir=log_dir)
            command = CheckPointManager.get_tensorboard_command(log_dir)
            print(f"TensorBoard logs will be saved to: {log_dir}")
            print("To visualize logs in real time, run the following command:")
            print(command)

            url = CheckPointManager.get_tensorboard_url()
            print(f"Access TensorBoard at: {url}")
            
            command_file = os.path.join(self._run_dir, 'tensorboard_command.txt')
            with open(command_file, 'w') as f:
                f.write(command+'\n')
                f.write(f"Access TensorBoard at: {url}\n")
        return self._summary_writer

class CkptNode:
    def __init__(self, root_dir, id, branch_id, name, epoch, iter, parent_id, next_id, comment=''):
        self._root_dir = root_dir
        self.id = id
        self.branch_id = branch_id
        self.name = name
        self.epoch = epoch
        self.iter = iter
        self.parent_id = parent_id
        self.next_id = next_id
        self.comment = comment
        self.child_ids = [] # List of child checkpoints (branch, ckpt_name)

    def __repr__(self):
        return (f"CkptNode(id={self.id}, branch_id={self.branch_id}, "
                f"epoch={self.epoch}, iter={self.iter}, "
                f"parent_id={self.parent_id}, next_id={self.next_id})")

    @property
    def dir(self):
        return os.path.join(self._root_dir, str(self.branch_id), str(self.id))

    @property
    def is_latest(self):
        return self.next_id is None
    
    @property
    def is_base(self):
        return self.parent_id == -1

    @property
    def is_branch(self):
        return CkptNode.is_branch_node(self.id)

    def add_child(self, child_id):
        self.child_ids.append(child_id)

    @staticmethod
    def is_branch_node(branch_id:int) -> bool:
        return str(branch_id).startswith("1")
    
    @staticmethod
    def is_ckpt_node(ckpt_id:int) -> bool:
        return str(ckpt_id).startswith("2")
    

    @staticmethod
    def is_topk_node(topk_id:int) -> bool:
        return str(topk_id).startswith("3")
    
    @staticmethod
    def convert_save_data(epoch:int, iter:int, model:ModelBase, optimizer: torch.optim.Optimizer = None, scheduler: torch.optim.lr_scheduler._LRScheduler = None, optimizer_info = None, extra_data=None) -> dict:
        save_data = {
            'epoch': epoch,
            'iter': iter,
            'model_config': model.get_params(),
            'model_state_dict': model.state_dict(),
            'optimizer_info': optimizer_info,
            'optimizer_state_dict': None,
            'scheduler_state_dict': None,
            'extra_data': extra_data,
        }
        if optimizer is not None:
            save_data['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler is not None:
            save_data['scheduler_state_dict'] = scheduler.state_dict()
        return save_data

    def save_model(self, model:ModelBase, epoch: int, iter: int, optimizer: torch.optim.Optimizer = None, scheduler: torch.optim.lr_scheduler._LRScheduler = None, optimizer_info = None, extra_data=None, epoch_zero_fill=4, iter_zero_fill=8) -> 'CkptNode':
        # Save the model and optimizer at the specified save_epoch
        ckpt_name = CkptNode._generate_ckpt_name_from_epoch_iter(epoch, iter, epoch_zero_fill, iter_zero_fill)
        
        os.makedirs(self.dir, exist_ok=True)
        model_file_path = os.path.join(self.dir, "model.pt")
        save_data = CkptNode.convert_save_data(epoch, iter, model, optimizer, scheduler, optimizer_info, extra_data)
        torch.save(save_data, model_file_path)

        return self.dir

    def load_model(self, func_create_model_form_config, device=None, extra_config_data=None, is_print=True) -> Tuple[ModelBase, int, int, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler, dict, dict]:
        load_path = os.path.join(self.dir, "model.pt")
        if is_print:
            print(f"Load model from {load_path}")

        checkpoint = torch.load(load_path, weights_only=False)
        model_config:dict = checkpoint['model_config']
        if extra_config_data is not None:
            model_config.update(extra_config_data)
        model: ModelBase = func_create_model_form_config(model_config)
        model.load_state_dict(checkpoint['model_state_dict'])

        if device is not None:
            model.to(device)

        optimizer = None
        scheduler = None
        optimizer_info = None
        if checkpoint.get('optimizer_info') is not None and checkpoint.get('optimizer_state_dict') is not None:
            optimizer_info = checkpoint['optimizer_info']
            optimizer: torch.optim.Optimizer = get_optimizer(model, optimizer_info, is_return_scheduler=False)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler: torch.optim.lr_scheduler._LRScheduler = get_scheduler(optimizer, optimizer_info.get('scheduler', None))
            if checkpoint.get('scheduler_state_dict') is not None and scheduler is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        global_step = checkpoint['iter']

        extra_data = checkpoint['extra_data']

        return model, epoch, global_step, optimizer, scheduler, optimizer_info, extra_data
    
    @staticmethod
    def _generate_ckpt_name_from_epoch_iter(epoch: int, iter: int, epoch_zero_fill=4, iter_zero_fill=8) -> str:
        epoch_str = str(epoch).zfill(epoch_zero_fill)
        iter_str = str(iter).zfill(iter_zero_fill)
        return f"model-e-{epoch_str}-i-{iter_str}.pt"
