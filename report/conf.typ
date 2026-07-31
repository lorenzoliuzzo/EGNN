#import "@preview/touying:0.6.1": *


#let title-slide(..args) = touying-slide-wrapper(
	self => {
		let info = self.info + args.named()
		
		let body = {
			set align(center + horizon)
			if info.title != none { 
				block(
					fill: self.colors.primary,
					width: 80%,
					inset: (y: 1em),
					radius: 1em,
					text(
						size: 1.5em, 
						fill: self.colors.neutral-lightest, 
						weight: "bold", 
						info.title
					),
				)
			}

			set text(fill: self.colors.neutral-darkest)
			if info.author != none { 
				block(info.author) 
			}
			if info.date != none { 
				block(utils.display-info-date(self))
			}
		}

		touying-slide(self: self, body)
	}
)


#let slide(title: auto, ..args) = touying-slide-wrapper(
	self => {
		let header-content = {
			set align(top)
			show: components.cell.with(fill: self.colors.primary, inset: 1em)

			set align(horizon)
			set text(fill: self.colors.neutral-lightest, size: 1.5em)
			if title != auto {
			  utils.call-or-display(self, title)
			} else {
			  utils.display-current-heading(level: 2)
			}
		}

		let footer-content = {
			set align(bottom)
			show: components.cell.with(fill: self.colors.primary, inset: 1em)
			set align(horizon)
			set text(
				fill: self.colors.neutral-lightest, 
				size: .8em
			)
			utils.call-or-display(self, self.info.title)
			h(1fr)
			context utils.slide-counter.display() + " / " + utils.last-slide-number
		}
	
		let conf = config-page(
			header: header-content, 
			footer: footer-content
		)
	  
		touying-slide(self: utils.merge-dicts(self, conf), ..args)
	}
)


#let new-section-slide(self: none, body) = touying-slide-wrapper(
	self => {
		let main-body = {
			set align(center + horizon)
			set text(
				size: 2.3em, 
				fill: self.colors.primary, 
				weight: "bold", 
				style: "italic"
			)
			utils.display-current-heading(level: 1)
		}
		touying-slide(self: self, main-body)
	}
)


#let focus-slide(body) = touying-slide-wrapper(
	self => {
		set text(fill: self.colors.neutral-lightest, size: 2em)
		let config = config-page(fill: self.colors.primary, margin: 2.3em)
		touying-slide(
			self: utils.merge-dicts(self, config), 
			align(horizon + center, body)
		)
	}
)


#let bamboo-theme(aspect-ratio: "16-9", ..args, body) = {	
	show: touying-slides.with(
		config-page(
			paper: "presentation-" + aspect-ratio,
			margin: (top: 5.2em, bottom: 3em, x: 2em),
		),
		config-colors(
			primary: rgb("#efca6d"),
			neutral-lightest: rgb("#ffffff"),
			neutral-darkest: rgb("#000000"),
		),
		config-methods(alert: utils.alert-with-primary-color),
		config-common(
			slide-fn: slide, 
			new-section-slide-fn: new-section-slide
		),	
		config-info(..args)
    )
	
	set text(size: 20pt)
	body
}