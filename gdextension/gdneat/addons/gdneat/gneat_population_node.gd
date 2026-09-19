class_name GDNeatPopulationNode extends Node

@export var config : GDNeatConfig
@onready var population := _make_population()

func pre_iterate() -> void:
	pass

func iterate(genomes: Array[GDNeatGenome]) -> void:
	pass

func post_iterate() -> void:
	pass

func _make_population() -> GDNeatPopulation:
	var pop := GDNeatPopulation.new()
	pop.create(config)
	return pop

func _process(_delta: float) -> void:
	pre_iterate()
	population.step(func(genomes: Array[GDNeatGenome]):
		iterate(genomes)
	)
	post_iterate()
