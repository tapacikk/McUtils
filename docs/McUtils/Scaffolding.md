# <a id="McUtils.Scaffolding">McUtils.Scaffolding</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/__init__.py#L1)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/__init__.py#L1?message=Update%20Docs)]
</div>
    
Provides development utilities.
Each utility attempts to be almost entirely standalone (although there is
a small amount of cross-talk within the packages).
In order of usefulness, the design is:
1. `Logging` provides a flexible logging interface where the log data can be
reparsed and loggers can be passed around
2. `Serializers`/`Checkpointing` provides interfaces for writing/loading data
to file and allows for easy checkpoint loading
3. `Jobs` provides simpler interfaces for running jobs using the existing utilities
4. `CLIs` provides simple command line interface helpers

### Members
<div class="container alert alert-secondary bg-light">
  <div class="row">
   <div class="col" markdown="1">
[Cache](Scaffolding/Caches/Cache.md)   
</div>
   <div class="col" markdown="1">
[MaxSizeCache](Scaffolding/Caches/MaxSizeCache.md)   
</div>
   <div class="col" markdown="1">
[ObjectRegistry](Scaffolding/Caches/ObjectRegistry.md)   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
[PseudoPickler](Scaffolding/Serializers/PseudoPickler.md)   
</div>
   <div class="col" markdown="1">
[BaseSerializer](Scaffolding/Serializers/BaseSerializer.md)   
</div>
   <div class="col" markdown="1">
[JSONSerializer](Scaffolding/Serializers/JSONSerializer.md)   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
[NumPySerializer](Scaffolding/Serializers/NumPySerializer.md)   
</div>
   <div class="col" markdown="1">
[NDarrayMarshaller](Scaffolding/Serializers/NDarrayMarshaller.md)   
</div>
   <div class="col" markdown="1">
[HDF5Serializer](Scaffolding/Serializers/HDF5Serializer.md)   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
[YAMLSerializer](Scaffolding/Serializers/YAMLSerializer.md)   
</div>
   <div class="col" markdown="1">
[ModuleSerializer](Scaffolding/Serializers/ModuleSerializer.md)   
</div>
   <div class="col" markdown="1">
[Schema](Scaffolding/Schema/Schema.md)   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
[Logger](Scaffolding/Logging/Logger.md)   
</div>
   <div class="col" markdown="1">
[NullLogger](Scaffolding/Logging/NullLogger.md)   
</div>
   <div class="col" markdown="1">
[LogLevel](Scaffolding/Logging/LogLevel.md)   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
[LogParser](Scaffolding/Logging/LogParser.md)   
</div>
   <div class="col" markdown="1">
[Checkpointer](Scaffolding/Checkpointing/Checkpointer.md)   
</div>
   <div class="col" markdown="1">
[CheckpointerKeyError](Scaffolding/Checkpointing/CheckpointerKeyError.md)   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
[DumpCheckpointer](Scaffolding/Checkpointing/DumpCheckpointer.md)   
</div>
   <div class="col" markdown="1">
[JSONCheckpointer](Scaffolding/Checkpointing/JSONCheckpointer.md)   
</div>
   <div class="col" markdown="1">
[NumPyCheckpointer](Scaffolding/Checkpointing/NumPyCheckpointer.md)   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
[HDF5Checkpointer](Scaffolding/Checkpointing/HDF5Checkpointer.md)   
</div>
   <div class="col" markdown="1">
[DictCheckpointer](Scaffolding/Checkpointing/DictCheckpointer.md)   
</div>
   <div class="col" markdown="1">
[NullCheckpointer](Scaffolding/Checkpointing/NullCheckpointer.md)   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
[PersistenceLocation](Scaffolding/Persistence/PersistenceLocation.md)   
</div>
   <div class="col" markdown="1">
[PersistenceManager](Scaffolding/Persistence/PersistenceManager.md)   
</div>
   <div class="col" markdown="1">
[BaseObjectManager](Scaffolding/ObjectBackers/BaseObjectManager.md)   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
[FileBackedObjectManager](Scaffolding/ObjectBackers/FileBackedObjectManager.md)   
</div>
   <div class="col" markdown="1">
[Config](Scaffolding/Configurations/Config.md)   
</div>
   <div class="col" markdown="1">
[ParameterManager](Scaffolding/Configurations/ParameterManager.md)   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
[Job](Scaffolding/Jobs/Job.md)   
</div>
   <div class="col" markdown="1">
[JobManager](Scaffolding/Jobs/JobManager.md)   
</div>
   <div class="col" markdown="1">
[CLI](Scaffolding/CLIs/CLI.md)   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
[CommandGroup](Scaffolding/CLIs/CommandGroup.md)   
</div>
   <div class="col" markdown="1">
[Command](Scaffolding/CLIs/Command.md)   
</div>
   <div class="col" markdown="1">
   
</div>
</div>
</div>













---


<div markdown="1" class="text-secondary">
<div class="container">
  <div class="row">
   <div class="col" markdown="1">
**Feedback**   
</div>
   <div class="col" markdown="1">
**Examples**   
</div>
   <div class="col" markdown="1">
**Templates**   
</div>
   <div class="col" markdown="1">
**Documentation**   
</div>
   <div class="col" markdown="1">
   
</div>
   <div class="col" markdown="1">
   
</div>
   <div class="col" markdown="1">
   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
[Bug](https://github.com/McCoyGroup/McUtils/issues/new?title=Documentation%20Improvement%20Needed)/[Request](https://github.com/McCoyGroup/McUtils/issues/new?title=Example%20Request)   
</div>
   <div class="col" markdown="1">
[Edit](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Scaffolding.md)/[New](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Scaffolding.md)   
</div>
   <div class="col" markdown="1">
[Edit](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Scaffolding.md)/[New](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Scaffolding.md)   
</div>
   <div class="col" markdown="1">
[Edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/__init__.py#L1?message=Update%20Docs)   
</div>
   <div class="col" markdown="1">
   
</div>
   <div class="col" markdown="1">
   
</div>
   <div class="col" markdown="1">
   
</div>
</div>
</div>
</div>